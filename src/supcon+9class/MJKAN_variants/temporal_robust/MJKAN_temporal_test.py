# ============================================================================
# STANDALONE TEST — load combined_temporal model FROM HF, reproduce all KPIs.
# This is the reproducibility-video eval: published model in, KPIs out.
# ============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from huggingface_hub import hf_hub_download
from pathlib import Path
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"mjkan_temporal")
Path(OUT).mkdir(parents=True, exist_ok=True)
CLASSES=["audio_streaming","cdn_web_assets","ecommerce","file_transfer","gaming",
         "messaging","search_info_news","social_media","video_streaming"]
NC=len(CLASSES)

# ---- model defs (must match training exactly) ----
class RSWAF(nn.Module):
    def __init__(s,ng=8,gmin=-2.,gmax=2.):
        super().__init__();g=torch.linspace(gmin,gmax,ng);s.register_buffer("grid",g);s.inv=nn.Parameter(torch.tensor(1.0/((gmax-gmin)/(ng-1))))
    def forward(s,x): return 1.0-torch.tanh((x.unsqueeze(-1)-s.grid)*s.inv)**2
class FasterKANLayer(nn.Module):
    def __init__(s,i,o,ng=8):
        super().__init__();s.rswaf=RSWAF(ng);s.ln=nn.LayerNorm(i);s.spline=nn.Linear(i*ng,o);s.base=nn.Linear(i,o)
    def forward(s,x): xn=s.ln(x);return s.spline(s.rswaf(xn).reshape(xn.size(0),-1))+s.base(xn)
class FasterKANReasoning(nn.Module):
    def __init__(s,d,ng=8): super().__init__();s.l1=FasterKANLayer(d,d,ng);s.bn=nn.BatchNorm1d(d);s.l2=FasterKANLayer(d,d,ng)
    def forward(s,x): return s.l2(F.relu(s.bn(s.l1(x))))
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
        for tt in range(L):h=a[:,tt]*h+bx[:,tt];ys.append((h*Cm[:,tt].unsqueeze(1)).sum(-1))
        return s.out_proj((torch.stack(ys,1)+s.D*xc)*F.silu(z))
class Encoder(nn.Module):
    def __init__(s,nf=5,dm=128,nl=2,ds=16,dp=0.1):
        super().__init__();s.embed=nn.Linear(nf,dm)
        s.blocks=nn.ModuleList([SSMBlock(dm,ds) for _ in range(nl)])
        s.norms=nn.ModuleList([nn.LayerNorm(dm) for _ in range(nl)]);s.drop=nn.Dropout(dp);s.onorm=nn.LayerNorm(dm)
    def forward(s,seq):
        mk=(seq.abs().sum(-1)>0).float();x=s.embed(seq)
        for b,n in zip(s.blocks,s.norms):x=x+s.drop(b(n(x),mk))
        x=s.onorm(x);mm=mk.unsqueeze(-1);return (x*mm).sum(1)/mm.sum(1).clamp(min=1.0)
class GatedFiLM(nn.Module):
    def __init__(s,n,dm,h=64):
        super().__init__();s.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,2*dm))
        s.gate=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,dm),nn.Sigmoid());s.dm=dm
    def forward(s,h,c):g=s.net(c);gm,bt=g[:,:s.dm],g[:,s.dm:];gt=s.gate(c);return gt*((1+gm)*h+bt)+(1-gt)*h
class MJKAN(nn.Module):
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__();s.encoder=Encoder(nf,dm);s.film=GatedFiLM(nc,dm)
        s.kan=FasterKANReasoning(dm);s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,x,c): h=s.film(s.encoder(x),c); r=h+s.kan(h); return h,F.normalize(s.proj(r),dim=1)

# ---- LOAD model + norm ----
if os.getenv("MJKAN_FROM_HF") == "1":
    ck=torch.load(hf_hub_download(HF,"combined_temporal/combined_temporal_best.pt"),map_location=DEVICE,weights_only=False)
else:
    ck=torch.load(Path(OUT,"combined_temporal_best.pt"),map_location=DEVICE,weights_only=False)
norm=ck["norm"]; sm,ss,cm,cs=norm["sm"],norm["ss"],norm["cm"],norm["cs"]
N_CTX=len(cm)
model=MJKAN(5,N_CTX).to(DEVICE); model.load_state_dict(ck["model"]); model.eval()
print(f"[loaded combined_temporal from HF — saved acc_quic={ck.get('acc_quic'):.3f}, acc_tls={ck.get('acc_tls'):.3f}]")

# ---- load test data (same splits) ----
q=np.load(hf_hub_download(HF,"combined_temporal/data/quic_temporal.npz"),allow_pickle=True)
t=np.load(hf_hub_download(HF,"combined_temporal/data/tls_temporal.npz"),allow_pickle=True)
q_seq,q_ctx,q_y,q_week=q["seq"],q["ctx"],q["label"].astype(int),q["week"].astype(int)
t_seq,t_ctx,t_y,t_month=t["seq"],t["ctx"],t["label"].astype(int),t["month"].astype(int)
q_tr=np.isin(q_week,[44,45]); q_te=np.isin(q_week,[46,47])
t_tr=t_month<=9; t_te=t_month>=10

def norm_seq(S): mm=(np.abs(S).sum(-1,keepdims=True)>0); return ((S-sm)/ss*mm).astype(np.float32)
def norm_ctx(C): return ((C-cm)/cs).astype(np.float32)
@torch.no_grad()
def embed(sq,cx,bs=1024):
    Z=[]
    for i in range(0,len(sq),bs):
        Z.append(model(torch.from_numpy(norm_seq(sq[i:i+bs])).float().to(DEVICE),
                       torch.from_numpy(norm_ctx(cx[i:i+bs])).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T;idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)

# reference bank = TRAIN embeddings (balanced)
TR_seq=np.concatenate([q_seq[q_tr],t_seq[t_tr]]); TR_ctx=np.concatenate([q_ctx[q_tr],t_ctx[t_tr]]); TR_y=np.concatenate([q_y[q_tr],t_y[t_tr]])
ztr=embed(TR_seq,TR_ctx); rng=np.random.default_rng(0)
ri=np.concatenate([rng.choice(np.where(TR_y==c)[0],min(4000,(TR_y==c).sum()),replace=False) for c in range(NC)])
RZ,RY=ztr[ri],TR_y[ri]

# ---- KPI 1: temporal generalization ----
zq=embed(q_seq[q_te],q_ctx[q_te]); pq=knn(RZ,RY,zq); acc_q=(pq==q_y[q_te]).mean()
zt=embed(t_seq[t_te],t_ctx[t_te]); pt=knn(RZ,RY,zt); acc_t=(pt==t_y[t_te]).mean()
# ---- KPI 2: embedding quality ----
allz=np.concatenate([zq,zt]); ally=np.concatenate([q_y[q_te],t_y[t_te]])
cents=np.stack([RZ[RY==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
intra=np.mean([(allz[ally==c]@cents[c]/ (np.linalg.norm(allz[ally==c],axis=1)+1e-9)).mean() for c in range(NC) if (ally==c).sum()])
inter=(cents@cents.T)[np.triu_indices(NC,1)].mean()
# ---- KPI 3: latency ----
z1=torch.from_numpy(norm_seq(q_seq[:1])).float().to(DEVICE); c1=torch.from_numpy(norm_ctx(q_ctx[:1])).float().to(DEVICE)
for _ in range(20):
    with torch.no_grad(): _=model(z1,c1)
if DEVICE=="cuda": torch.cuda.synchronize()
ts=[]
for _ in range(200):
    a=time.perf_counter()
    with torch.no_grad(): _=model(z1,c1)
    if DEVICE=="cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter()-a)*1000)

print(f"\n{'='*55}\nREPRODUCED KPIs (from published HF model)\n{'='*55}")
print(f"  Accuracy QUIC (temporal):  {acc_q:.3f}")
print(f"  Accuracy TLS  (temporal):  {acc_t:.3f}")
print(f"  Embedding intra-class:     {intra:.3f}  (target > 0.7)")
print(f"  Embedding inter-class:     {inter:.3f}  (target < 0.3)")
print(f"  Latency (median):          {np.median(ts):.2f} ms  (target < 100)")
print(f"\n  Per-class accuracy (QUIC / TLS):")
for c in range(NC):
    aq=(pq[q_y[q_te]==c]==c).mean() if (q_y[q_te]==c).sum() else float('nan')
    at=(pt[t_y[t_te]==c]==c).mean() if (t_y[t_te]==c).sum() else float('nan')
    print(f"    {CLASSES[c]:18s}: "+(f"{aq:.3f}" if not np.isnan(aq) else "  -  ")+" / "+(f"{at:.3f}" if not np.isnan(at) else "  -  "))
print(f"\n  [matches training-time numbers? saved: {ck.get('acc_quic'):.3f}/{ck.get('acc_tls'):.3f}]")

# ---- save results ----
per_class = {}
for c in range(NC):
    aq=(pq[q_y[q_te]==c]==c).mean() if (q_y[q_te]==c).sum() else None
    at=(pt[t_y[t_te]==c]==c).mean() if (t_y[t_te]==c).sum() else None
    per_class[CLASSES[c]] = {"quic": float(aq) if aq is not None else None,
                              "tls":  float(at) if at is not None else None}
result = {
    "model": "mjkan_temporal",
    "acc_quic": float(acc_q), "acc_tls": float(acc_t),
    "intra_cosine": float(intra), "inter_cosine": float(inter),
    "latency_ms_median": float(np.median(ts)),
    "per_class": per_class,
}
out_json = Path(OUT, "mjkan_temporal_eval.json")
out_json.write_text(json.dumps(result, indent=2))
print(f"\n[saved] {out_json}")

# ---- t-SNE plot (cosine metric on balanced sample; caption computed on same sample) ----
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    COLORS = ["#E59500","#B5654A","#7A8C5E","#A0763E","#C75D4A",
              "#5B7B8A","#8A6FA0","#C98AA0","#2B6E6A"]
    PERCLASS = 600
    rng2 = np.random.default_rng(0)
    keep = np.concatenate([rng2.choice(np.where(ally==c)[0], min(PERCLASS,(ally==c).sum()), replace=False)
                           for c in range(NC) if (ally==c).sum()>0])
    Zs, ys = allz[keep], ally[keep]
    print(f"t-SNE on {len(Zs)} points ({PERCLASS}/class target)...")
    emb2d = TSNE(n_components=2, metric="cosine", init="pca",
                 perplexity=30, learning_rate="auto", random_state=0).fit_transform(Zs)
    # compute intra/inter on the SAME sample so caption matches the plot
    present = [c for c in range(NC) if (ys==c).sum()>0]
    C = np.stack([Zs[ys==c].mean(0) for c in present]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    intra_s = np.mean([(Zs[ys==c]@C[i]).mean() for i,c in enumerate(present)])
    inter_s = (C@C.T)[np.triu_indices(len(present),1)].mean()
    plt.figure(figsize=(10,8), dpi=150)
    plt.gca().set_facecolor("#FBF7F2"); plt.gcf().patch.set_facecolor("#FBF7F2")
    for i, c in enumerate(present):
        m = ys==c
        plt.scatter(emb2d[m,0], emb2d[m,1], s=10, alpha=0.7,
                    color=COLORS[c], label=CLASSES[c].replace("_"," "), edgecolors="none")
    plt.title("MJKAN Temporal — flagship embedding space (z), temporal test set",
              fontsize=13, color="#2B2118", pad=12)
    plt.figtext(0.5, 0.012,
                f"intra-class cosine {intra_s:.3f}   ·   inter-class cosine {inter_s:+.3f}   ·   {len(Zs)} flows",
                ha="center", fontsize=10, color="#7A6A5C")
    plt.xticks([]); plt.yticks([])
    for sp in plt.gca().spines.values(): sp.set_visible(False)
    plt.legend(loc="center left", bbox_to_anchor=(1.0,0.5), frameon=False, fontsize=9, markerscale=1.6)
    plt.tight_layout(rect=[0,0.03,1,1])
    out_tsne = str(Path(OUT, "tsne_embeddings.png"))
    plt.savefig(out_tsne, bbox_inches="tight", facecolor="#FBF7F2")
    plt.close()
    print(f"[saved] {out_tsne}")
except Exception as e:
    print(f"[tsne skipped: {e}]")
