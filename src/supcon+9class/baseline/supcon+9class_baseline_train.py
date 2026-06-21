# ============================================================================
# SHIGAN / MJKAN-Net — RUNG 3 (BEHAVIORAL 9-CLASS): SUPERVISED CONTRASTIVE
# Same proven recipe (BN proj head, LR=1e-3 warmup->constant, tau=0.1) on the
# new 9-class behavioral data. Class imbalance handled via WEIGHTED SAMPLER
# (correct for SupCon) using class_weights from meta.json. New HF folder.
# ============================================================================
import json, time, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------------------- CONFIG ---------------------------
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, HfApi, login
HF_REPO="donbosoc/shigan-mjkan-baseline"; HF_FOLDER="behavioral9+supcon"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"baseline"); Path(OUT).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUT,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
SEED=0
EPOCHS=50; BATCH=512; LR=1e-3; WD=1e-4; TAU=0.1; WARMUP=2
D_MODEL=128; N_LAYERS=2; D_STATE=16; DROPOUT=0.1
KNN_EVERY=2; KNN_K=10; KNN_VAL_PER_CLASS=1000
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
# ----------------------------------------------------------------
torch.manual_seed(SEED); np.random.seed(SEED)
meta=json.loads(Path(hf_hub_download(HF_REPO,f"{HF_FOLDER}/data/meta.json")).read_text()); NC=meta["num_classes"]; CATS=meta["categories"]
CLASS_W=meta.get("class_weights",{c:1.0 for c in CATS})    # <<< from Phase 2
print("[device]",DEVICE,"| classes:",CATS,"| NC:",NC)

class DS(Dataset):
    def __init__(s,split):
        d=np.load(hf_hub_download(HF_REPO,f"{HF_FOLDER}/data/{split}.npz")); s.seq=torch.from_numpy(d["seq"]).float(); s.y=torch.from_numpy(d["label"]).long()
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.seq[i], s.y[i]
train_ds=DS("train"); val_ds=DS("val")

# ---- WEIGHTED SAMPLER: oversample rare classes (ecommerce) for balanced batches ----
y_tr=train_ds.y.numpy()
per_sample_w=np.array([CLASS_W[CATS[c]] for c in y_tr],dtype=np.float64)
sampler=WeightedRandomSampler(weights=torch.from_numpy(per_sample_w),
                              num_samples=len(y_tr), replacement=True)
train_ld=DataLoader(train_ds,batch_size=BATCH,sampler=sampler,num_workers=2,
                    pin_memory=(DEVICE=="cuda"),drop_last=True)
print("[sampler] weighted; class_weights:",{k:round(v,2) for k,v in CLASS_W.items()})

# ---- model (verbatim from your trained recipe) ----
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

def supcon_loss(z,labels,tau=TAU):
    sim=z@z.T/tau; B=z.size(0)
    sm=torch.eye(B,dtype=torch.bool,device=z.device); sim=sim.masked_fill(sm,-1e9)
    lab=labels.view(-1,1); pos=(lab==lab.T)&~sm
    logp=sim-torch.logsumexp(sim,1,keepdim=True)
    pc=pos.sum(1).clamp(min=1); loss=-(logp*pos).sum(1)/pc
    v=pos.sum(1)>0; return loss[v].mean()

model=SupConModel(meta["n_seq_feats"],D_MODEL,N_LAYERS,D_STATE,DROPOUT).to(DEVICE)
print("[model] encoder params:",sum(p.numel() for p in model.encoder.parameters()))
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
def lr_lambda(ep): return (ep+1)/WARMUP if ep<WARMUP else 1.0
sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

rng=np.random.default_rng(SEED)
def subset(ds,per):
    y=ds.y.numpy(); idx=np.concatenate([rng.choice(np.where(y==c)[0],min(per,(y==c).sum()),replace=False) for c in range(NC)])
    return ds.seq[idx].numpy(), y[idx]
val_seq,val_y=subset(val_ds,KNN_VAL_PER_CLASS); tr_seq,tr_y=subset(train_ds,500)
@torch.no_grad()
def embed_z(arr):     # <<< eval on z (your deployed embedding), not h
    model.eval(); out=[]
    for i in range(0,len(arr),1024):
        xb=torch.from_numpy(arr[i:i+1024]).float().to(DEVICE)
        out.append(model(xb)[1].cpu().numpy())   # z
    return np.concatenate(out)
def knn_acc():
    ref=embed_z(tr_seq); q=embed_z(val_seq)
    sims=q@ref.T; tk=np.argsort(-sims,1)[:,:KNN_K]
    pred=np.array([np.bincount(tr_y[tk[i]],minlength=NC).argmax() for i in range(len(q))])
    return (pred==val_y).mean()

best=-1
for ep in range(1,EPOCHS+1):
    model.train(); t0=time.time(); tl=0; nb=0; neg=0; ns=0
    for seq,y in train_ld:
        seq,y=seq.to(DEVICE),y.to(DEVICE)
        h,z=model(seq); loss=supcon_loss(z,y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl+=loss.item(); nb+=1
        if nb<=5:
            with torch.no_grad():
                sim=z@z.T; lab=y.view(-1,1); sm=torch.eye(len(y),dtype=torch.bool,device=z.device)
                neg+=sim[(lab!=lab.T)].mean().item(); ns+=1
    sched.step()
    log={"epoch":ep,"supcon_loss":tl/nb,"neg_sim":neg/ns,"lr":opt.param_groups[0]["lr"]}
    msg=f"[ep {ep:02d}] loss={tl/nb:.4f} neg_sim={neg/ns:.3f} lr={opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s)"
    if ep%KNN_EVERY==0 or ep==EPOCHS:
        ka=knn_acc(); log["val_knn_acc"]=ka; msg+=f" val_knn={ka:.4f}"
        if ka>best:
            best=ka
            torch.save({"model":model.state_dict(),"encoder":model.encoder.state_dict(),
                        "meta":meta,"tau":TAU,"val_knn":ka},Path(OUT,"supcon_best.pt"))
            msg+=" *"
    print(msg)
print(f"\n[done] best val k-NN = {best:.4f}")
if os.getenv("MJKAN_PUSH_HF") == "1":
    api=HfApi()
    api.upload_file(path_or_fileobj=str(Path(OUT,"supcon_best.pt")),
                    path_in_repo=f"{HF_FOLDER}/supcon_best.pt",repo_id=HF_REPO)
    print(f"[hf] pushed -> https://huggingface.co/{HF_REPO}/tree/main/{HF_FOLDER}")