# ============================================================================
# MJKAN (Option 2 — deck-faithful): SSM -> GatedFiLM -> FasterKAN REASONING
#   -> Projection head -> z (SupCon). KAN reasoning + projection are SEPARATE.
# ============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from huggingface_hub import hf_hub_download, login, HfApi
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
api=HfApi()
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; torch.backends.cudnn.benchmark=True
HF="donbosoc/shigan-mjkan-baseline"; OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"mjkan"); Path(OUT).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUT,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
EPOCHS,LR,WARMUP,TAU,BATCH=50,1e-3,3,0.1,512; SEP_W=0.5

meta=json.load(open(hf_hub_download(HF,"protocol_invariance/data/joint_meta.json")))
CATS=meta["categories"]; NC=meta["num_classes"]; N_CTX=meta["n_ctx_feats"]
dtr=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_train.npz"),allow_pickle=True)
dte=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_test.npz"),allow_pickle=True)
tr_seq,tr_ctx,tr_y=dtr["seq"],dtr["ctx"],dtr["label"].astype(int)
te_seq,te_ctx,te_y=dte["seq"],dte["ctx"],dte["label"].astype(int); te_p=dte["protocol"].astype(str)

# ---- FasterKAN reasoning (RSWAF basis) ----
class RSWAF(nn.Module):
    def __init__(s,num_grids=8,gmin=-2.0,gmax=2.0):
        super().__init__(); g=torch.linspace(gmin,gmax,num_grids)
        s.register_buffer("grid",g); s.inv=nn.Parameter(torch.tensor(1.0/((gmax-gmin)/(num_grids-1))))
    def forward(s,x): return 1.0-torch.tanh((x.unsqueeze(-1)-s.grid)*s.inv)**2
class FasterKANLayer(nn.Module):
    def __init__(s,i,o,num_grids=8):
        super().__init__();s.rswaf=RSWAF(num_grids);s.ln=nn.LayerNorm(i)
        s.spline=nn.Linear(i*num_grids,o);s.base=nn.Linear(i,o)
    def forward(s,x):
        xn=s.ln(x); b=s.rswaf(xn).reshape(xn.size(0),-1); return s.spline(b)+s.base(xn)
class FasterKANReasoning(nn.Module):
    """Reasoning stage: nonlinear KAN transform of the conditioned embedding."""
    def __init__(s,dim,num_grids=8):
        super().__init__();s.l1=FasterKANLayer(dim,dim,num_grids);s.bn=nn.BatchNorm1d(dim);s.l2=FasterKANLayer(dim,dim,num_grids)
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
        nn.init.zeros_(s.net[-1].weight);nn.init.zeros_(s.net[-1].bias)
    def forward(s,h,c):g=s.net(c);gm,bt=g[:,:s.dm],g[:,s.dm:];gt=s.gate(c);return gt*((1+gm)*h+bt)+(1-gt)*h

class MJKAN(nn.Module):
    """Deck-faithful: encoder -> GatedFiLM -> FasterKAN reasoning -> projection -> z."""
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__()
        s.encoder=Encoder(nf,dm)
        s.film=GatedFiLM(nc,dm)
        s.kan=FasterKANReasoning(dm)                      # <-- reasoning stage
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))  # <-- projection head (kept)
    def forward(s,x,c):
        h=s.film(s.encoder(x),c)        # encode + condition
        r=s.kan(h)                      # KAN reasoning
        r=h+r                           # residual: reasoning refines the embedding (stability)
        z=s.proj(r)                     # projection head -> contrastive space
        return h, F.normalize(z,dim=1)

def supcon(z,y,tau=TAU):
    sim=z@z.T/tau; sim=sim-sim.max(1,keepdim=True).values.detach()
    y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
    lm=torch.ones_like(pos); lm.fill_diagonal_(0)
    exp=torch.exp(sim)*lm; logp=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    base=-((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()
    cents=[F.normalize(z[(y.squeeze()==c)].mean(0),dim=0) for c in y.unique() if (y.squeeze()==c).sum()>0]
    if len(cents)>1:
        C=torch.stack(cents); inter=(C@C.T).triu(1); sep=inter[inter!=0].mean()
    else: sep=torch.tensor(0.,device=z.device)
    return base+SEP_W*sep

class DS(Dataset):
    def __len__(s): return len(tr_y)
    def __getitem__(s,i): return torch.from_numpy(tr_seq[i]).float(),torch.from_numpy(tr_ctx[i]).float(),int(tr_y[i])
cnt=np.bincount(tr_y,minlength=NC); cw=(cnt.sum()/(NC*np.clip(cnt,1,None))).astype(np.float32)
sampler=WeightedRandomSampler(torch.from_numpy(cw[tr_y]).double(),len(tr_y),True)
dl=DataLoader(DS(),batch_size=BATCH,sampler=sampler,num_workers=2,drop_last=True,pin_memory=True)
model=MJKAN(meta["n_seq_feats"],N_CTX).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
print(f"MJKAN params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M (KAN reasoning + projection separate)")
print(f"{'='*55}\nMJKAN deck-faithful — joint ({EPOCHS}ep)\n{'='*55}")
for ep in range(EPOCHS):
    for g in opt.param_groups: g["lr"]=LR*(ep+1)/WARMUP if ep<WARMUP else LR
    model.train();tot=0;nb=0
    for seq,ctx,yb in dl:
        seq,ctx,yb=seq.to(DEVICE,non_blocking=True),ctx.to(DEVICE,non_blocking=True),yb.to(DEVICE,non_blocking=True)
        _,z=model(seq,ctx); loss=supcon(z,yb)
        if torch.isnan(loss): print("  NaN skip"); continue
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();tot+=loss.item();nb+=1
    if (ep+1)%5==0 or ep==EPOCHS-1: print(f"  ep{ep+1}: loss={tot/nb:.4f}")

@torch.no_grad()
def embed(sq,cx,bs=1024):
    model.eval();Z=[]
    for i in range(0,len(sq),bs):
        Z.append(model(torch.from_numpy(sq[i:i+bs]).float().to(DEVICE),torch.from_numpy(cx[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T;idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)
ztr=embed(tr_seq,tr_ctx); zte=embed(te_seq,te_ctx)
rng=np.random.default_rng(0); ri=np.concatenate([rng.choice(np.where(tr_y==c)[0],min(4000,(tr_y==c).sum()),replace=False) for c in range(NC)])
pred=knn(ztr[ri],tr_y[ri],zte)
qa=(pred[te_p=="quic"]==te_y[te_p=="quic"]).mean(); ta=(pred[te_p=="tls"]==te_y[te_p=="tls"]).mean()
cents=np.stack([ztr[tr_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
inter=(cents@cents.T)[np.triu_indices(NC,1)].mean(); intra=np.mean([(zte[te_y==c]@cents[c]).mean() for c in range(NC)])
z1=torch.from_numpy(te_seq[:1]).float().to(DEVICE);c1=torch.from_numpy(te_ctx[:1]).float().to(DEVICE)
for _ in range(20):
    with torch.no_grad(): _=model(z1,c1)
if DEVICE=="cuda": torch.cuda.synchronize()
ts=[]
for _ in range(200):
    a=time.perf_counter()
    with torch.no_grad(): _=model(z1,c1)
    if DEVICE=="cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter()-a)*1000)
print(f"\n{'='*55}\nMJKAN — KPI SCORECARD\n{'='*55}")
print(f"  QUIC: {qa:.3f}   TLS: {ta:.3f}")
print(f"  intra: {intra:.3f}   inter: {inter:.3f}   latency: {np.median(ts):.2f}ms")
print(f"  vs V7 (SSM+MLP): QUIC 0.877 / TLS 0.737 / inter 0.234")
for c in range(NC):
    m=te_y==c; print(f"    {CATS[c]:18s}: {(pred[m]==c).mean():.3f}")
torch.save({"model":model.state_dict(),"meta":meta,"acc_quic":float(qa),"acc_tls":float(ta)},f"{OUT}/mjkan_best.pt")
if os.getenv("MJKAN_PUSH_HF") == "1":
    api.upload_file(path_or_fileobj=f"{OUT}/mjkan_best.pt",path_in_repo="mjkan/mjkan_best.pt",repo_id=HF)
    print("\n[MJKAN pushed to HF]")
else:
    print("\n[MJKAN saved locally — use push_to_hf.py mjkan to upload]")