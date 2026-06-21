# ============================================================================
# V5z — DANN attacking the Z EMBEDDING (the deployed representation).
# Differs from V5 (h-attack) only in WHERE the discriminator/GRL attach: z, not h.
# Same stability: no AMP, gentle lambda, grad clip, NaN guard, collapse guard.
# Goal: dom_acc(z) drifts down while QUIC-test stays ~0.85; does TLS move?
# ============================================================================
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from huggingface_hub import hf_hub_download
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF_REPO="donbosoc/shigan-mjkan-baseline"
OUTDIR=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"v5_dann"); Path(OUTDIR).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUTDIR,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
BATCH, EPOCHS, LR, TAU, SEED = 512, 15, 2e-4, 0.1, 0
LAMBDA_MAX, GAMMA, CLIP = 0.3, 5.0, 5.0
torch.manual_seed(SEED); np.random.seed(SEED)

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lambd): ctx.lambd=lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.lambd, None
def grl(x,lambd): return GRL.apply(x,lambd)

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
class DomainHead(nn.Module):
    def __init__(s,d=128,h=128):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),
                                                nn.Dropout(0.2),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)

def supcon_loss(z,y,tau=TAU):
    z=F.normalize(z,dim=1); sim=z@z.T/tau
    sim=sim-sim.max(1,keepdim=True).values.detach()
    y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
    lm=torch.ones_like(pos); lm.fill_diagonal_(0)
    exp=torch.exp(sim)*lm
    logp=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    mpp=(pos*logp).sum(1)/pos.sum(1).clamp(min=1)
    return -mpp.mean()

def load_v1():
    ck=torch.load(hf_hub_download(HF_REPO,"protocol_invariance/v1_joint/supcon_v1_best.pt"),
                  map_location=DEVICE,weights_only=False)
    m=SupConModel(ck["meta"]["n_seq_feats"],128,2,16,0.1).to(DEVICE); m.load_state_dict(ck["model"])
    return m, ck["meta"]
model, meta = load_v1()
CATS=meta["categories"]; NC=meta["num_classes"]; CW=np.array(meta["class_weights"],np.float32)
ECOM=CATS.index("ecommerce")
dom=DomainHead(128).to(DEVICE)   # input dim 128 = z dim (same as h dim here)
print(f"[V1] loaded for DANN-on-Z | ecommerce idx={ECOM} (excluded from domain loss)")

class DS(Dataset):
    def __init__(s,sp):
        d=np.load(hf_hub_download(HF_REPO,f"protocol_invariance/data/joint_{sp}.npz"),allow_pickle=True)
        s.seq=d["seq"]; s.y=d["label"].astype(int); s.proto=d["protocol"].astype(str)
    def __len__(s): return len(s.y)
    def __getitem__(s,i):
        return (torch.from_numpy(s.seq[i]).float(), int(s.y[i]),
                1.0 if s.proto[i]=="tls" else 0.0,
                0.0 if s.y[i]==ECOM else 1.0)
tr=DS("train"); te=DS("test")
w=CW[tr.y]; sampler=WeightedRandomSampler(torch.from_numpy(w).double(),len(w),True)
tl=DataLoader(tr,batch_size=BATCH,sampler=sampler,num_workers=2,drop_last=True)

opt=torch.optim.AdamW(list(model.parameters())+list(dom.parameters()),lr=LR,weight_decay=1e-4)
def lambda_at(p): return (2.0/(1.0+np.exp(-GAMMA*p))-1.0)*LAMBDA_MAX

@torch.no_grad()
def embed_z(seq,bs=512):
    model.eval(); Z=[]
    for i in range(0,len(seq),bs):
        Z.append(model(torch.from_numpy(seq[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T; idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)
def evalp(tag):
    ztr=embed_z(tr.seq); zte=embed_z(te.seq)
    rng=np.random.default_rng(0)
    ri=np.concatenate([rng.choice(np.where(tr.y==c)[0],min(2000,(tr.y==c).sum()),replace=False) for c in range(NC)])
    pred=knn(ztr[ri],tr.y[ri],zte)
    q=te.proto=="quic"; t=te.proto=="tls"
    ov=(pred==te.y).mean(); qa=(pred[q]==te.y[q]).mean(); ta=(pred[t]==te.y[t]).mean()
    print(f"  [{tag}] overall={ov:.3f} | QUIC={qa:.3f} | TLS={ta:.3f} <<<")
    for c in range(NC):
        m=t&(te.y==c)
        if m.sum(): print(f"      {CATS[c]:18s} {(pred[m]==c).mean():.3f}")
    return ov,qa,ta

best=0; steps=EPOCHS*len(tl); gstep=0; nan_batches=0
for ep in range(EPOCHS):
    model.train(); dom.train(); tot=0; dl=0; dacc=0; nb=0
    for seq,y,dy,dm_mask in tl:
        seq,y,dy,dm_mask=seq.to(DEVICE),y.to(DEVICE),dy.to(DEVICE),dm_mask.to(DEVICE)
        lam=lambda_at(gstep/steps); gstep+=1
        opt.zero_grad()
        h,z=model(seq)
        l_sup=supcon_loss(z,y)
        dlogit=dom(grl(z,lam))                         # <<< ATTACK Z (was h)
        l_dom_all=F.binary_cross_entropy_with_logits(dlogit,dy,reduction="none")
        l_dom=(l_dom_all*dm_mask).sum()/dm_mask.sum().clamp(min=1)
        loss=l_sup+l_dom
        if not torch.isfinite(loss): nan_batches+=1; continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        torch.nn.utils.clip_grad_norm_(dom.parameters(),CLIP)
        opt.step()
        tot+=l_sup.item(); dl+=l_dom.item()
        with torch.no_grad():
            pr=(torch.sigmoid(dlogit)>0.5).float()
            dacc+=(((pr==dy).float()*dm_mask).sum()/dm_mask.sum().clamp(min=1)).item()
        nb+=1
    print(f"[ep{ep+1}] supcon={tot/max(nb,1):.3f} dom_loss={dl/max(nb,1):.3f} "
          f"dom_acc={dacc/max(nb,1):.3f} lambda={lam:.3f} nan_skips={nan_batches}")
    if (ep+1)%3==0 or ep==EPOCHS-1:
        ov,qa,ta=evalp(f"ep{ep+1}")
        if qa<0.5:
            print(f"   !! QUIC collapsed ({qa:.3f}) — lower LAMBDA_MAX to 0.1, re-run."); break
        if ta>best:
            best=ta; torch.save({"model":model.state_dict(),"meta":meta,"epoch":ep+1,
                "tls_acc":ta,"quic_acc":qa,"overall":ov,"dom_acc":dacc/max(nb,1),"attack":"z"},
                f"{OUTDIR}/v5z_dann_best.pt")
            print(f"   ** saved (TLS={ta:.3f}, QUIC={qa:.3f}, dom_acc={dacc/max(nb,1):.3f})")
print(f"\nV5z (DANN-on-Z) DONE. Best TLS={best:.3f} (V1=0.72, V5-h=0.727).")