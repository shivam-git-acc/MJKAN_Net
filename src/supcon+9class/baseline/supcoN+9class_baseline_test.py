# ============================================================================
# 9-CLASS BEHAVIORAL MODEL — FULL EVALUATION
# Per-class k-NN, confusion matrix, intra/inter cosine, t-SNE. Eval on z.
# ============================================================================
import json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF_REPO="donbosoc/shigan-mjkan-baseline"; HF_FOLDER="behavioral9+supcon"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"baseline")
if os.getenv("MJKAN_FROM_HF") == "1":
    CKPT=hf_hub_download(HF_REPO,f"{HF_FOLDER}/supcon_best.pt")
else:
    CKPT=str(Path(OUT,"supcon_best.pt"))
KNN_K=10; PER_CLASS_EVAL=2000; REF_PER_CLASS=1000

# ---- model defs (verbatim) ----
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

ckpt=torch.load(CKPT,map_location=DEVICE,weights_only=False)
meta=ckpt["meta"]; CATS=meta["categories"]; NC=meta["num_classes"]
model=SupConModel(meta["n_seq_feats"],128,2,16,0.1).to(DEVICE)
model.load_state_dict(ckpt["model"]); model.eval()
print("[loaded] 9-class | val_knn@train:",round(ckpt.get("val_knn",0),4))

@torch.no_grad()
def embed_z(arr):
    out=[]
    for i in range(0,len(arr),1024):
        xb=torch.from_numpy(arr[i:i+1024]).float().to(DEVICE)
        out.append(model(xb)[1].cpu().numpy())
    return np.concatenate(out)

# ---- load test + balanced sample ----
d=np.load(hf_hub_download(HF_REPO,f"{HF_FOLDER}/data/test.npz")); seq=d["seq"]; y=d["label"]
rng=np.random.default_rng(0)
qi=np.concatenate([rng.choice(np.where(y==c)[0],min(PER_CLASS_EVAL,(y==c).sum()),replace=False) for c in range(NC)])
ri=np.concatenate([rng.choice(np.where(y==c)[0],min(REF_PER_CLASS,(y==c).sum()),replace=False) for c in range(NC)])
zq=embed_z(seq[qi]); yq=y[qi]; zr=embed_z(seq[ri]); yr=y[ri]
print(f"[embedded] query {zq.shape} ref {zr.shape}")

# ---- k-NN predict (exclude self via ref/query split) ----
sims=zq@zr.T; tk=np.argsort(-sims,1)[:,:KNN_K]
pred=np.array([np.bincount(yr[tk[i]],minlength=NC).argmax() for i in range(len(zq))])
overall=(pred==yq).mean()
print(f"\n[overall k-NN acc] {overall:.4f}")
print("\n[per-class k-NN accuracy]")
for c in range(NC):
    m=yq==c; acc=(pred[m]==c).mean() if m.sum() else 0
    print(f"   {CATS[c]:18s} {acc:.4f}")

# ---- confusion matrix ----
cm=np.zeros((NC,NC),int)
for t,p in zip(yq,pred): cm[t,p]+=1
cmn=cm/cm.sum(1,keepdims=True)
print("\n[confusion matrix] (row=true, col=pred, normalized)")
print("            "+" ".join(f"{c[:6]:>7s}" for c in CATS))
for i in range(NC):
    print(f"{CATS[i]:11s} "+" ".join(f"{cmn[i,j]:7.2f}" for j in range(NC)))

# ---- intra/inter class cosine (KPI metrics, on z) ----
cents=np.stack([zq[yq==c].mean(0) for c in range(NC)])
cents/=np.linalg.norm(cents,axis=1,keepdims=True)
intra=np.array([ (zq[yq==c]@cents[c]).mean() for c in range(NC) ])  # mean cos to own centroid
inter=cents@cents.T; np.fill_diagonal(inter,np.nan)
print("\n[intra-class cosine] (higher=tighter, KPI target >0.7)")
for c in range(NC): print(f"   {CATS[c]:18s} {intra[c]:.3f}")
print(f"\n[mean intra] {intra.mean():.3f}")
print(f"[mean inter-centroid cosine] {np.nanmean(inter):.3f} (KPI target <0.3)")
print(f"[max inter pair] {np.nanmax(inter):.3f} between "
      f"{CATS[np.nanargmax(inter)//NC]} & {CATS[np.nanargmax(inter)%NC]}")

# ---- t-SNE ----
try:
    from sklearn.manifold import TSNE; import matplotlib.pyplot as plt
    ids=np.concatenate([np.where(yq==c)[0][:400] for c in range(NC)])
    emb=TSNE(n_components=2,perplexity=30,init="pca",random_state=0).fit_transform(zq[ids])
    plt.figure(figsize=(10,8))
    for c in range(NC):
        m=yq[ids]==c
        plt.scatter(emb[m,0],emb[m,1],s=12,alpha=0.6,label=CATS[c])
    plt.legend(markerscale=2,fontsize=8,loc="best")
    plt.title(f"9-class behavioral z-embeddings (k-NN={overall:.3f})")
    plt.tight_layout(); plt.savefig(str(Path(OUT,"behavioral9_tsne.png")),dpi=130); plt.show()
    print("[saved] behavioral9_tsne.png")
except Exception as e: print("[tsne skipped]",e)
