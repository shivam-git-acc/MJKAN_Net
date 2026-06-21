# ============================================================================
# RUNG 3 — FINAL z-SPACE VISUALS (t-SNE + cosine matrix on z, the deployed embedding)
# Pulls SupCon model from HF. Saves + pushes plots to baseline+supcon/.
# ============================================================================
import json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, HfApi, login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

HF_REPO="donbosoc/shigan-mjkan-baseline"; HF_FOLDER="baseline+supcon"; HF_DATA="baseline6classdata"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"baseline_6class"); Path(OUT).mkdir(parents=True,exist_ok=True)
PER_CLASS_PLOT=400; PER_CLASS_SIM=1500
BASELINE_INTRA=0.695; BASELINE_INTER=0.623
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

CKPT=hf_hub_download(HF_REPO,f"{HF_FOLDER}/supcon_best.pt")
ck=torch.load(CKPT,map_location=DEVICE,weights_only=False); meta=ck["meta"]; CATS=meta["categories"]; NC=meta["num_classes"]

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
    def forward(s,seq):
        h=s.encoder(seq); z=F.normalize(s.proj(h),dim=1); return h,z
model=SupConModel(meta["n_seq_feats"]).to(DEVICE); model.load_state_dict(ck["model"]); model.eval()

d=np.load(hf_hub_download(HF_REPO,f"{HF_DATA}/pool_test.npz")); seq=d["seq"]; y=d["label"]; rng=np.random.default_rng(0)
@torch.no_grad()
def embed_z(arr):
    out=[]
    for i in range(0,len(arr),1024):
        xb=torch.from_numpy(arr[i:i+1024]).float().to(DEVICE)
        out.append(model(xb)[1].cpu().numpy())   # z, already L2-normalized
    return np.concatenate(out)

# ---- cosine matrix (z) ----
sidx=np.concatenate([rng.choice(np.where(y==c)[0],min(PER_CLASS_SIM,(y==c).sum()),replace=False) for c in range(NC)])
E=embed_z(seq[sidx]); Lb=y[sidx]
means=np.stack([E[Lb==c].mean(0) for c in range(NC)]); means/=np.linalg.norm(means,axis=1,keepdims=True)+1e-9
M=means@means.T
def intra(c):
    e=E[Lb==c]; idx=rng.choice(len(e),min(300,len(e)),replace=False); e=e[idx]
    s=e@e.T; iu=np.triu_indices(len(e),1); return float(s[iu].mean())
intra_vals={CATS[c]:round(intra(c),3) for c in range(NC)}
mean_intra=round(float(np.mean(list(intra_vals.values()))),3)
mean_inter=round(float(M[~np.eye(NC,dtype=bool)].mean()),3)
print(f"[z] mean intra={mean_intra} (baseline {BASELINE_INTRA}) | mean inter={mean_inter} (baseline {BASELINE_INTER})")

fig1,ax=plt.subplots(figsize=(7,6))
im=ax.imshow(M,cmap="viridis",vmin=-0.3,vmax=1.0)
ax.set_xticks(range(NC));ax.set_yticks(range(NC));ax.set_xticklabels(CATS,rotation=45,ha="right");ax.set_yticklabels(CATS)
for i in range(NC):
    for j in range(NC): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="white" if M[i,j]<0.6 else "black",fontsize=8)
ax.set_title(f"Inter-class cosine — SupCon (z)\nintra={mean_intra} inter={mean_inter}")
fig1.colorbar(im);fig1.tight_layout();p_mat=Path(OUT,"cosine_matrix_supcon_z.png");fig1.savefig(p_mat,dpi=130);plt.close(fig1)

# ---- t-SNE (z) ----
pidx=np.concatenate([rng.choice(np.where(y==c)[0],min(PER_CLASS_PLOT,(y==c).sum()),replace=False) for c in range(NC)])
Ep=embed_z(seq[pidx]); Lp=y[pidx]
print("[t-SNE on z] computing...")
T=TSNE(n_components=2,init="pca",perplexity=30,max_iter=1000,random_state=0).fit_transform(Ep)
fig2,ax=plt.subplots(figsize=(8,7));cmap=plt.cm.tab10
for c in range(NC):
    m=Lp==c;ax.scatter(T[m,0],T[m,1],s=6,alpha=0.6,color=cmap(c),label=CATS[c])
ax.legend(markerscale=2,fontsize=9);ax.set_title("Embedding t-SNE — SupCon (z)")
ax.set_xticks([]);ax.set_yticks([]);fig2.tight_layout()
p_tsne=Path(OUT,"tsne_supcon_z.png");fig2.savefig(p_tsne,dpi=130);plt.close(fig2)

metrics=dict(space="z",mean_intra=mean_intra,mean_inter=mean_inter,intra_per_class=intra_vals,
             inter_matrix={CATS[i]:{CATS[j]:round(float(M[i,j]),3) for j in range(NC)} for i in range(NC)},
             baseline_intra=BASELINE_INTRA,baseline_inter=BASELINE_INTER)
p_json=Path(OUT,"supcon_z_metrics.json");p_json.write_text(json.dumps(metrics,indent=2))
print("saved:",p_mat.name,p_tsne.name,p_json.name)

if os.getenv("MJKAN_PUSH_HF") == "1":
    api=HfApi()
    for f in (p_mat,p_tsne,p_json):
        api.upload_file(path_or_fileobj=str(f),path_in_repo=f"{HF_FOLDER}/{f.name}",repo_id=HF_REPO)
    print(f"[hf] pushed z-visuals -> {HF_REPO}/{HF_FOLDER}")