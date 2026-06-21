# ============================================================================
# HELD-OUT SUPCON (Cell B): k-NN on z. Support+query = all 6 classes.
# File sharing was NEVER trained on. Compare held-out acc vs baseline-CE (77%).
# ============================================================================
import json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download

HF_REPO="donbosoc/shigan-mjkan-baseline"; HF_DATA="baseline6classdata"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"held_out_6class")
K=10; SUPPORT_PER_CLASS=50; QUERY_PER_CLASS=2000
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
if os.getenv("MJKAN_FROM_HF") == "1":
    ck=torch.load(hf_hub_download(HF_REPO,"baseline+supcon/supcon_heldout.pt"),map_location=DEVICE,weights_only=False)
else:
    ck=torch.load(Path(OUT,"supcon_heldout.pt"),map_location=DEVICE,weights_only=False)
meta=ck["meta"]; ALL=meta["categories"]; HELDOUT=ck["heldout"]; ho_idx=ALL.index(HELDOUT); NC=len(ALL)

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
def emb_z(arr):
    out=[]
    for i in range(0,len(arr),1024):
        xb=torch.from_numpy(arr[i:i+1024]).float().to(DEVICE)
        out.append(model(xb)[1].cpu().numpy())
    return np.concatenate(out)

sup_s,sup_l,q_s,q_l=[],[],[],[]
for c in range(NC):
    idx=np.where(y==c)[0]; rng.shuffle(idx)
    si=idx[:SUPPORT_PER_CLASS]; qi=idx[SUPPORT_PER_CLASS:SUPPORT_PER_CLASS+QUERY_PER_CLASS]
    sup_s.append(seq[si]); sup_l.append(np.full(len(si),c))
    q_s.append(seq[qi]); q_l.append(np.full(len(qi),c))
sup_e=emb_z(np.concatenate(sup_s)); sup_l=np.concatenate(sup_l)
q_e=emb_z(np.concatenate(q_s)); q_l=np.concatenate(q_l)
sims=q_e@sup_e.T; tk=np.argsort(-sims,1)[:,:K]
pred=np.array([np.bincount(sup_l[tk[i]],minlength=NC).argmax() for i in range(len(q_e))])
print(f"[SupCon HELD-OUT] overall k-NN acc (all 6, trained on 5) = {(pred==q_l).mean():.4f}")
print("(baseline cross-entropy held-out was: overall 0.819, File sharing 0.774, Streaming 0.682)\n")
for ci,cat in enumerate(ALL):
    m=q_l==ci; tag=" <-- HELD OUT (never trained)" if ci==ho_idx else ""
    print(f"  {cat:18s} {(pred[m]==ci).mean():.3f} (n={m.sum()}){tag}")
