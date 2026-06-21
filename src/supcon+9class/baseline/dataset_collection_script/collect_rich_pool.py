# ============================================================================
# SHIGAN / MJKAN-Net — 9-CLASS BEHAVIORAL DATASET COLLECTION
# Phase 1: reads CESNET-QUIC22 CSVs, maps APP tags → 9 behavioral categories.
# Phase 2: caps large classes, fits normalization on train, saves pool_*.npz
#           with norm stats + class_weights in pool_meta.json.
#
# Usage:
#   CESNET_DATA_DIR=/path/to/cesnet-quic22  python3 collect_rich_pool.py
#   CESNET_DATA_DIR=...  MJKAN_PUSH_HF=1    python3 collect_rich_pool.py
#
# CESNET-QUIC22: https://data.cesnet.cz/research/datasets/cesnet-quic22
# ============================================================================
import ast, glob, json, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def _kaggle_secret(key):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key) or ""
    except Exception:
        return ""

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _kaggle_secret("HF_TOKEN")
from huggingface_hub import HfApi, login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------------
# Defaults are the Kaggle-mounted paths; override with env vars for local runs.
ROOT      = os.getenv("CESNET_DATA_DIR",  "/kaggle/input/datasets/anishanandhan/cesnet/cesnet-quic22")
OUT       = os.path.join(os.getenv("DATA_DIR", "/kaggle/working"), "behavioral9classdata")
HF_REPO   = os.getenv("HF_DATASET_REPO", "") or _kaggle_secret("HF_DATASET_REPO")
HF_FOLDER = "behavioral9classdata"

ROWS_PER_FILE    = 150_000
COLLECT_CAP      = 40_000
CHUNK            = 50_000
TRAIN_WEEKS      = ["W-2022-44","W-2022-45"]
VAL_WEEKS        = ["W-2022-46"]
TEST_WEEKS       = ["W-2022-47"]
SEED             = 0
TARGET_PER_CLASS = 40_000   # cap — large classes trimmed, small kept whole
BALANCE_EVAL     = False    # keep val/test natural; use class_weights instead
EPS              = 1e-6
# ----------------------------------------------------------------------------
rng     = np.random.default_rng(SEED)
SEQ_LEN, N_SEQ, N_CTX, JGAIN = 30, 5, 6, 16.0
SEQ_NAMES = ["packet_size","direction","IAT","jitter_rfc3550","dir_change"]
CTX_NAMES = ["flow_duration","packet_rate","roundtrip_density","up_down_ratio","truncation_ratio","burstiness"]
CTX_LOG   = {"flow_duration","packet_rate"}

TAG2CAT = {}
def _add(cat, tags):
    for t in tags: TAG2CAT[t] = cat
_add("video_streaming",  ["youtube","facebook-media","spanbang","xhamster"])
_add("audio_streaming",  ["spotify","shazam","playradio"])
_add("messaging",        ["whatsapp","facebook-messenger","discord"])
_add("social_media",     ["facebook-web","instagram","tiktok","snapchat","dcard"])
_add("file_transfer",    ["google-drive","google-docs","google-photos","google-colab","facebook-rupload"])
_add("cdn_web_assets",   ["google-fonts","fontawesome","jsdelivr","cloudflare-cdnjs","google-gstatic","google-usercontent"])
_add("ecommerce",        ["alza-www","dm-de","drmax","rohlik","ebay-kleinanzeigen","goout","revolut"])
_add("gaming",           ["unitygames","blitz-gg","easybrain","chess-com","csgo-market","gamedock"])
_add("search_info_news", ["google-www","kaggle","ncbi-gov","mdpi","livescore","medium","blogger","4chan","forum24","sme-sk"])

KEEP    = sorted(set(TAG2CAT.values()))
NC      = len(KEEP)
cat2idx = {c:i for i,c in enumerate(KEEP)}

Path(OUT).mkdir(parents=True, exist_ok=True)

# ============================================================
# PHASE 1 — Collect raw flows from CESNET CSVs
# ============================================================
def _norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())

def detect_cols(df):
    cn = {c:_norm(c) for c in df.columns}
    def f(*nm):
        for x in nm:
            for o,n in cn.items():
                if n==_norm(x): return o
        return None
    c = dict(ppi=f("PPI"),app=f("APP"),duration=f("DURATION"),bytes=f("BYTES"),
             bytes_rev=f("BYTES_REV"),packets=f("PACKETS"),packets_rev=f("PACKETS_REV"),
             ppi_len=f("PPI_LEN"),roundtrips=f("PPI_ROUNDTRIPS"))
    miss = [k for k,v in c.items() if v is None]
    if miss: raise ValueError(f"Missing cols {miss}. Found {list(df.columns)}")
    return c

def parse_ppi(v):
    if isinstance(v,(list,tuple)): a=list(v)
    elif isinstance(v,str):
        s=v.strip()
        if not s or s in ("[]","nan","None"): return [],[],[]
        try: a=ast.literal_eval(s)
        except: return [],[],[]
    else: return [],[],[]
    if isinstance(a,list) and len(a)==3 and all(isinstance(x,list) for x in a): return a[0],a[1],a[2]
    return [],[],[]

def build_seq(ti,di,si):
    t=np.asarray(ti,float).ravel(); d=np.asarray(di,float).ravel(); s=np.asarray(si,float).ravel()
    n=min(len(t),len(d),len(s),SEQ_LEN); out=np.zeros((SEQ_LEN,N_SEQ),np.float32)
    if n==0: return out
    t,d,s=t[:n],np.sign(d[:n]),np.abs(s[:n]); d[d==0]=1.0
    iat=t.copy(); iat[0]=0.0; iat=np.clip(iat,0,None)
    jit=np.zeros(n)
    for i in range(1,n): jit[i]=jit[i-1]+(abs(iat[i]-iat[i-1])-jit[i-1])/JGAIN
    dch=np.zeros(n); dch[1:]=(d[1:]!=d[:-1])
    out[:n,0]=s; out[:n,1]=d; out[:n,2]=iat; out[:n,3]=jit; out[:n,4]=dch
    return out

def build_ctx(ti,row,c):
    x=np.zeros(N_CTX,np.float32)
    dur=float(row[c["duration"]]); by=float(row[c["bytes"]]); byr=float(row[c["bytes_rev"]])
    pk=float(row[c["packets"]]); pkr=float(row[c["packets_rev"]]); pl=float(row[c["ppi_len"]])
    rt=float(row[c["roundtrips"]]); tot=pk+pkr
    iat=np.asarray(ti,float).ravel()
    if iat.size: iat=iat.copy(); iat[0]=0.0; iat=np.clip(iat,0,None)
    x[0]=dur; x[1]=tot/(dur+EPS); x[2]=rt/(pl+EPS); x[3]=by/(by+byr+EPS)
    x[4]=pl/(tot+EPS); x[5]=(iat.std()/(iat.mean()+EPS)) if iat.size>1 else 0.0
    return x

def split_of(wk):
    if wk in TRAIN_WEEKS: return "train"
    if wk in VAL_WEEKS:   return "val"
    if wk in TEST_WEEKS:  return "test"
    return None

files = []
for p in sorted(glob.glob(os.path.join(ROOT,"W-*","*","flows-*.csv"))):
    wk = next((x for x in Path(p).parts if x.startswith("W-")),"W-?")
    files.append((p,wk))
if not files:
    raise RuntimeError(
        f"No CESNET CSV files found under CESNET_DATA_DIR={ROOT}\n"
        f"Download from: https://data.cesnet.cz/research/datasets/cesnet-quic22"
    )
print(f"[phase 1] {len(files)} CSV files found")
cols = detect_cols(pd.read_csv(files[0][0], nrows=5))
use  = [cols[k] for k in ("ppi","app","duration","bytes","bytes_rev","packets","packets_rev","ppi_len","roundtrips")]

pools  = {sp:{c:{"seq":[],"ctx":[],"y":[],"tag":[]} for c in KEEP} for sp in ("train","val","test")}
counts = {sp:{c:0 for c in KEEP} for sp in ("train","val","test")}

for p,wk in files:
    sp = split_of(wk)
    if sp is None: continue
    read = 0
    for chunk in pd.read_csv(p, usecols=use, chunksize=CHUNK):
        if read >= ROWS_PER_FILE: break
        chunk = chunk[chunk[cols["app"]].isin(TAG2CAT)]
        for _,row in chunk.iterrows():
            tag = str(row[cols["app"]]).strip()
            cat = TAG2CAT.get(tag)
            if cat is None: continue
            if counts[sp][cat] >= COLLECT_CAP: continue
            ti,di,si = parse_ppi(row[cols["ppi"]])
            if min(len(ti),len(di),len(si)) < 2: continue
            pools[sp][cat]["seq"].append(build_seq(ti,di,si))
            pools[sp][cat]["ctx"].append(build_ctx(ti,row,cols))
            pools[sp][cat]["y"].append(cat2idx[cat])
            pools[sp][cat]["tag"].append(tag)
            counts[sp][cat] += 1
        read += len(chunk)
    print(f"  {sp} {wk}: {counts[sp]}")

def assemble(sp):
    S,X,Y,T = [],[],[],[]
    for c in KEEP:
        d = pools[sp][c]
        if d["y"]:
            S.append(np.stack(d["seq"])); X.append(np.stack(d["ctx"]))
            Y.append(np.asarray(d["y"])); T.extend(d["tag"])
    return (np.concatenate(S).astype(np.float32),
            np.concatenate(X).astype(np.float32),
            np.concatenate(Y).astype(np.int64),
            np.asarray(T))

raw = {}
for sp in ("train","val","test"):
    if any(pools[sp][c]["y"] for c in KEEP):
        raw[sp] = assemble(sp)
        dist = {KEEP[i]:int(n) for i,n in zip(*np.unique(raw[sp][2],return_counts=True))}
        print(f"[phase 1] {sp}: {len(raw[sp][2])} flows | {dist}")

# ============================================================
# PHASE 2 — Cap large classes + normalize
# ============================================================
def cap(seq, ctx, y, tag, target):
    keep_idx = []
    for c in range(NC):
        idx = np.where(y==c)[0]
        if len(idx) > target: idx = rng.choice(idx, target, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx); rng.shuffle(keep_idx)
    t = tag[keep_idx] if tag is not None else None
    return seq[keep_idx], ctx[keep_idx], y[keep_idx], t

str_, ctr, ytr, ttr = raw["train"]
counts_tr = np.bincount(ytr, minlength=NC)
print(f"\n[phase 2] train pool counts: {dict(zip(KEEP, counts_tr.tolist()))}")
str_, ctr, ytr, ttr = cap(str_, ctr, ytr, ttr, TARGET_PER_CLASS)
fc = np.bincount(ytr, minlength=NC)
print(f"[phase 2] after cap@{TARGET_PER_CLASS}: {dict(zip(KEEP,fc.tolist()))} -> {len(ytr)} flows")

# class weights (inverse-frequency, normalized to mean 1)
freq = fc / fc.sum()
w = 1.0 / (freq + EPS); w = w / w.mean()
class_weights = {KEEP[i]: float(w[i]) for i in range(NC)}
print(f"[phase 2] class_weights: {class_weights}")

# fit normalization on train only
log_idx  = [i for i,n in enumerate(CTX_NAMES) if n in CTX_LOG]
pmask    = np.any(str_ != 0, axis=2); valid = str_[pmask]
seq_mean = valid.mean(0); seq_std = valid.std(0) + EPS
cp       = ctr.copy(); cp[:,log_idx] = np.log1p(np.clip(cp[:,log_idx],0,None))
ctx_mean = cp.mean(0); ctx_std = cp.std(0) + EPS
print(f"[phase 2] seq_mean[:3]={seq_mean[:3].round(2).tolist()}")

def apply_norm(seq, ctx):
    m  = np.any(seq!=0, axis=2, keepdims=True)
    sn = ((seq - seq_mean) / seq_std) * m
    c  = ctx.copy(); c[:,log_idx] = np.log1p(np.clip(c[:,log_idx],0,None))
    cn = (c - ctx_mean) / ctx_std
    return sn.astype(np.float32), cn.astype(np.float32)

final_meta = dict(
    categories=KEEP, cat2idx=cat2idx, tag2cat=TAG2CAT,
    seq_feature_names=SEQ_NAMES, ctx_feature_names=CTX_NAMES,
    ctx_log_names=list(CTX_LOG),
    n_seq_feats=N_SEQ, n_ctx_feats=N_CTX,
    num_classes=NC, seq_len=SEQ_LEN,
    cap_per_class=TARGET_PER_CLASS,
    class_weights=class_weights,
    norm=dict(seq_mean=seq_mean.tolist(), seq_std=seq_std.tolist(),
              ctx_mean=ctx_mean.tolist(), ctx_std=ctx_std.tolist(),
              ctx_log_idx=log_idx)
)

saved = []
splits = {"train": (str_, ctr, ytr, ttr)}
for sp in ("val","test"):
    if sp in raw:
        s,x,y,t = raw[sp]
        if BALANCE_EVAL:
            s,x,y,t = cap(s,x,y,t,int(np.bincount(y,minlength=NC).min()))
        splits[sp] = (s,x,y,t)

for sp,(s,x,y,t) in splits.items():
    sn,cn = apply_norm(s,x)
    fp = Path(OUT, f"pool_{sp}.npz")
    save = dict(seq=sn, ctx=cn, label=y)
    if t is not None: save["tag"] = t
    np.savez_compressed(fp, **save)
    dist = {KEEP[i]:int(n) for i,n in zip(*np.unique(y,return_counts=True))}
    print(f"[save] pool_{sp}.npz  {sn.shape}  {dist}")
    saved.append(fp)

meta_fp = Path(OUT,"pool_meta.json")
meta_fp.write_text(json.dumps(final_meta, indent=2))
saved.append(meta_fp)
print(f"\nDONE — 9-class behavioral pools saved to {OUT}")

# ---- push to user's HF repo ----
if os.getenv("MJKAN_PUSH_HF") == "1":
    if not HF_REPO:
        print("[hf] skipped — set HF_DATASET_REPO=yourname/your-repo to push")
    else:
        api = HfApi()
        for f in saved:
            remote = f"{HF_FOLDER}/{f.name}"
            api.upload_file(path_or_fileobj=str(f), path_in_repo=remote, repo_id=HF_REPO)
            print(f"[hf] pushed {remote} -> {HF_REPO}")
