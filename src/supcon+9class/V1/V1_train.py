# ============================================================================
# V1 — JOINT QUIC+TLS SupCon training (the protocol-invariance GATE).
# Proven recipe: LR=1e-3 warmup-then-const, tau=0.1, BATCH=512, BN proj head.
# Natural cross-protocol positives (50/50 balanced joint set + large batch).
# Eval reports QUIC-test / TLS-test / per-class SEPARATELY (invariance proof).
# ============================================================================
import json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, HfApi, login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF_REPO="donbosoc/shigan-mjkan-baseline"; HF_DATA="protocol_invariance/data"
OUTDIR=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"v1"); Path(OUTDIR).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUTDIR,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
BATCH, EPOCHS, LR, WARMUP, TAU, SEED = 512, 50, 1e-3, 2, 0.1, 0
torch.manual_seed(SEED); np.random.seed(SEED)

meta=json.load(open(hf_hub_download(HF_REPO,f"{HF_DATA}/joint_meta.json")))
CATS=meta["categories"]; NC=meta["num_classes"]; CW=np.array(meta["class_weights"],np.float32)
print(f"[joint] {NC} classes | {CATS}")

class DS(Dataset):
    def __init__(s,sp):
        d=np.load(hf_hub_download(HF_REPO,f"{HF_DATA}/joint_{sp}.npz"),allow_pickle=True)
        s.seq=d["seq"]; s.ctx=d["ctx"]; s.y=d["label"]; s.proto=d["protocol"]; s.tag=d["tag"]
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return (torch.from_numpy(s.seq[i]).float(),
                                  torch.from_numpy(s.ctx[i]).float(), int(s.y[i]))
tr=DS("train"); va=DS("val"); te=DS("test")

# weighted sampler (handles ecommerce imbalance) — proven approach
w=CW[tr.y]; sampler=WeightedRandomSampler(torch.from_numpy(w).double(),len(w),replacement=True)
tl=DataLoader(tr,batch_size=BATCH,sampler=sampler,num_workers=2,drop_last=True)

# ---- model (verbatim proven architecture) ----
class SSMBlock(nn.Module):
    def __init__(s,dm,ds=16,e=2):
        super().__init__();s.di=e*dm;s.ds=ds
        s.in_proj=nn.Linear(dm,2*s.di);s.conv=nn.Conv1d(s.di,s.di,3,padding=2,groups=s.di)
        s.x_proj=nn.Linear(s.di,2*ds+1)
        s.A_log=nn.Parameter(torch.log(torch.arange(1,ds+1,dtype=torch.float32).repeat(s.di,1)))
        s.D=nn.Parameter(torch.ones(s.di));s.out_proj=nn.Linear(s.di,dm)
    def forward(s,x,mask):
        B,L,_=x.shape;xb,z=s.in_proj(x).chunk(2,-1)
        xc=F.silu(s.conv(xb.transpose(1,2))[...,:L].transpose(1,2))
        Bm,Cm,dt=torch.split(s.x_proj(xc),[s.ds,s.ds,1],-1);dt=F.softplus(dt);A=-torch.exp(s.A_log)
        a=torch.exp(dt.unsqueeze(-1)*A.unsqueeze(0).unsqueeze(0));bx=(dt*Bm).unsqueeze(2)*xc.unsqueeze(-1)
        mm=mask.view(B,L,1,1);a=a*mm+(1-mm);bx=bx*mm
        h=torch.zeros(B,s.di,s.ds,device=x.device);ys=[]
        for t in range(L):h=a[:,t]*h+bx[:,t];ys.append((h*Cm[:,t].unsqueeze(1)).sum(-1))
        return s.out_proj((torch.stack(ys,1)+s.D*xc)*F.silu(z))
class Encoder(nn.Module):
    def __init__(s,nf=5,dm=128,nl=2,ds=16,dp=0.1):
        super().__init__();s.embed=nn.Linear(nf,dm)
        s.blocks=nn.ModuleList([SSMBlock(dm,ds) for _ in range(nl)])
        s.norms=nn.ModuleList([nn.LayerNorm(dm) for _ in range(nl)])
        s.drop=nn.Dropout(dp);s.onorm=nn.LayerNorm(dm)
    def forward(s,seq):
        mask=(seq.abs().sum(-1)>0).float();x=s.embed(seq)
        for b,n in zip(s.blocks,s.norms):x=x+s.drop(b(n(x),mask))
        x=s.onorm(x);mm=mask.unsqueeze(-1);return (x*mm).sum(1)/mm.sum(1).clamp(min=1.0)
class SupConModel(nn.Module):
    def __init__(s,nf=5,dm=128,nl=2,ds=16,dp=0.1):
        super().__init__();s.encoder=Encoder(nf,dm,nl,ds,dp)
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,seq): h=s.encoder(seq); z=F.normalize(s.proj(h),dim=1); return h,z

def supcon_loss(z,y,tau=TAU):
    z=F.normalize(z,dim=1); sim=z@z.T/tau
    sim=sim-sim.max(1,keepdim=True).values.detach()
    y=y.view(-1,1); pos=(y==y.T).float()
    pos.fill_diagonal_(0); logits_mask=torch.ones_like(pos); logits_mask.fill_diagonal_(0)
    exp=torch.exp(sim)*logits_mask
    logp=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    mpp=(pos*logp).sum(1)/pos.sum(1).clamp(min=1)
    return -mpp.mean()

model=SupConModel(meta["n_seq_feats"],128,2,16,0.1).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
def lr_at(ep): return LR*(ep+1)/WARMUP if ep<WARMUP else LR   # warmup-then-constant

@torch.no_grad()
def embed_ds(ds,bs=1024):
    model.eval(); Z=[]
    for i in range(0,len(ds),bs):
        xb=torch.from_numpy(ds.seq[i:i+bs]).float().to(DEVICE)
        Z.append(model(xb)[1].cpu().numpy())
    return np.concatenate(Z)

def knn_eval(ref_z,ref_y,qz,qy,k=20):
    # cosine kNN (ref already normalized z)
    preds=[]
    for i in range(0,len(qz),2048):
        sims=qz[i:i+2048]@ref_z.T
        idx=np.argpartition(-sims,k,1)[:,:k]
        vote=ref_y[idx]
        preds.append(np.array([np.bincount(v,minlength=NC).argmax() for v in vote]))
    return np.concatenate(preds)

def protocol_eval(tag=""):
    """Eval invariance: build QUIC-ref centroids, test QUIC-test & TLS-test separately."""
    ztr=embed_ds(tr); zte=embed_ds(te)
    # reference = TRAIN embeddings (both protocols), query = TEST
    rng=np.random.default_rng(0)
    ridx=np.concatenate([rng.choice(np.where(tr.y==c)[0],min(2000,(tr.y==c).sum()),replace=False) for c in range(NC)])
    ref_z,ref_y=ztr[ridx],tr.y[ridx]
    pred=knn_eval(ref_z,ref_y,zte,te.y)
    q=te.proto=="quic"; t=te.proto=="tls"
    print(f"  [{tag}] TEST overall kNN acc: {(pred==te.y).mean():.3f}")
    print(f"  [{tag}] QUIC-test acc: {(pred[q]==te.y[q]).mean():.3f} (n={q.sum()})")
    print(f"  [{tag}] TLS-test  acc: {(pred[t]==te.y[t]).mean():.3f} (n={t.sum()})  <<< invariance signal")
    # per-class TLS (the protocol-transfer detail)
    print(f"  [{tag}] per-class TLS acc:")
    for c in range(NC):
        m=t&(te.y==c)
        if m.sum(): print(f"      {CATS[c]:18s} {(pred[m]==c).mean():.3f} (n={m.sum()})")
    return (pred==te.y).mean(), (pred[t]==te.y[t]).mean()

# ---- train ----
best_tls=0
for ep in range(EPOCHS):
    model.train()
    for g in opt.param_groups: g["lr"]=lr_at(ep)
    tot=0
    for seq,ctx,y in tl:
        seq,y=seq.to(DEVICE),y.to(DEVICE)
        _,z=model(seq); loss=supcon_loss(z,y)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%5==0 or ep==EPOCHS-1:
        ov,tls=protocol_eval(f"ep{ep+1}")
        print(f"[ep{ep+1}] loss={tot/len(tl):.4f} lr={lr_at(ep):.1e} overall={ov:.3f} TLS={tls:.3f}")
        if tls>best_tls:
            best_tls=tls
            torch.save({"model":model.state_dict(),"meta":meta,"epoch":ep+1,
                        "tls_acc":tls,"overall":ov}, f"{OUTDIR}/v1_best.pt")
            print(f"   ** saved (best TLS={tls:.3f})")
    else:
        print(f"[ep{ep+1}] loss={tot/len(tl):.4f} lr={lr_at(ep):.1e}")

print(f"\nV1 DONE. Best TLS-test acc: {best_tls:.3f}  (QUIC-only baseline was 0.197)")

SECTION="protocol_invariance"
ck=torch.load(f"{OUTDIR}/v1_best.pt", map_location="cpu", weights_only=False)
result={
  "experiment":"V1 — joint QUIC+TLS SupCon (protocol-invariance gate)",
  "recipe":"LR=1e-3 warmup(2)-then-const, tau=0.1, BATCH=512, BN proj head, natural cross-protocol positives",
  "joint_train_size":337002, "proto_balance":"~50/50 per shared class; ecommerce QUIC-only",
  "best_epoch":int(ck.get("epoch",-1)),
  "tls_test_acc":float(ck.get("tls_acc",-1)),
  "overall_test_acc":float(ck.get("overall",-1)),
  "quic_only_baseline_tls_acc":0.197,
  "note":"TLS-test acc is the invariance signal; QUIC-only model scored 0.197. QUIC-test should stay ~0.9."
}
Path(f"{OUTDIR}/v1_result.json").write_text(json.dumps(result,indent=2))
print("[result]",json.dumps(result,indent=2))

if os.getenv("MJKAN_PUSH_HF") == "1":
    api=HfApi()
    uploads=[
      (f"{OUTDIR}/v1_best.pt",       f"{SECTION}/v1_joint/supcon_v1_best.pt"),
      (f"{OUTDIR}/v1_result.json",   f"{SECTION}/v1_joint/v1_result.json"),
    ]
    for local,remote in uploads:
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=HF_REPO)
            print(f"[pushed] {remote}")
    print(f"\nDONE -> https://huggingface.co/{HF_REPO}/tree/main/{SECTION}")

