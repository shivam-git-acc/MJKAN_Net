# ============================================================================
# HELD-OUT SUPCON (Cell A): train SupCon on 5 classes, File sharing EXCLUDED.
# Same proven config: BatchNorm head, LR=1e-3, tau=0.1, warmup, 50 epochs.
# Saves supcon_heldout.pt locally (experiment — not pushed to main repo folder).
# ============================================================================
import json, time, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

HF_REPO="donbosoc/shigan-mjkan-baseline"; HF_DATA="baseline6classdata"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"held_out_6class"); Path(OUT).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUT,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
HELDOUT="File sharing"
EPOCHS=50; BATCH=512; LR=1e-3; WD=1e-4; TAU=0.1; WARMUP=2
D_MODEL=128; N_LAYERS=2; D_STATE=16; DROPOUT=0.1
KNN_EVERY=4; KNN_K=10; SEED=0
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
meta=json.loads(Path(hf_hub_download(HF_REPO,f"{HF_DATA}/pool_meta.json")).read_text())
ALL=meta["categories"]; ho_idx=ALL.index(HELDOUT)
keep_old=[i for i in range(len(ALL)) if i!=ho_idx]
old2new={o:n for n,o in enumerate(keep_old)}; NC5=len(keep_old)
print("[device]",DEVICE,"| training on:",[ALL[i] for i in keep_old],"| held out:",HELDOUT)

class DS(Dataset):
    def __init__(s,split):
        d=np.load(hf_hub_download(HF_REPO,f"{HF_DATA}/pool_{split}.npz")); y=d["label"]; m=y!=ho_idx
        s.seq=torch.from_numpy(d["seq"][m]).float()
        s.y=torch.tensor([old2new[int(v)] for v in y[m]]).long()
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.seq[i], s.y[i]
train_ds=DS("train"); val_ds=DS("val")
train_ld=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=(DEVICE=="cuda"),drop_last=True)

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

def supcon_loss(z,labels,tau=TAU):
    sim=z@z.T/tau; B=z.size(0)
    sm=torch.eye(B,dtype=torch.bool,device=z.device); sim=sim.masked_fill(sm,-1e9)
    lab=labels.view(-1,1); pos=(lab==lab.T)&~sm
    logp=sim-torch.logsumexp(sim,1,keepdim=True)
    pc=pos.sum(1).clamp(min=1); loss=-(logp*pos).sum(1)/pc
    v=pos.sum(1)>0; return loss[v].mean()

model=SupConModel(meta["n_seq_feats"],D_MODEL,N_LAYERS,D_STATE,DROPOUT).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
def lr_lambda(ep): return (ep+1)/WARMUP if ep<WARMUP else 1.0
sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

# val k-NN over the 5 TRAINED classes only (for checkpoint selection)
rng=np.random.default_rng(SEED)
def subset(ds,per):
    y=ds.y.numpy(); idx=np.concatenate([rng.choice(np.where(y==c)[0],min(per,(y==c).sum()),replace=False) for c in range(NC5)])
    return ds.seq[idx].numpy(), y[idx]
vs,vy=subset(val_ds,1000); ts,ty=subset(train_ds,500)
@torch.no_grad()
def emb_z(arr):
    model.eval(); out=[]
    for i in range(0,len(arr),1024):
        xb=torch.from_numpy(arr[i:i+1024]).float().to(DEVICE)
        out.append(model(xb)[1].cpu().numpy())
    return np.concatenate(out)
def knn5():
    ref=emb_z(ts); q=emb_z(vs); sims=q@ref.T; tk=np.argsort(-sims,1)[:,:KNN_K]
    pred=np.array([np.bincount(ty[tk[i]],minlength=NC5).argmax() for i in range(len(q))]); return (pred==vy).mean()

best=-1
for ep in range(1,EPOCHS+1):
    model.train(); t0=time.time(); tl=0; nb=0
    for seq,y in train_ld:
        seq,y=seq.to(DEVICE),y.to(DEVICE)
        h,z=model(seq); loss=supcon_loss(z,y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl+=loss.item(); nb+=1
    sched.step()
    msg=f"[ep {ep:02d}] loss={tl/nb:.4f} ({time.time()-t0:.0f}s)"
    if ep%KNN_EVERY==0 or ep==EPOCHS:
        ka=knn5(); msg+=f" val_knn5={ka:.4f}"
        if ka>best:
            best=ka; torch.save({"model":model.state_dict(),"encoder":model.encoder.state_dict(),
                "meta":meta,"heldout":HELDOUT,"old2new":old2new,"keep_old":keep_old},Path(OUT,"supcon_heldout.pt"))
            msg+=" *"
    print(msg)
print(f"\n[done] best val_knn5 = {best:.4f} -> saved supcon_heldout.pt")