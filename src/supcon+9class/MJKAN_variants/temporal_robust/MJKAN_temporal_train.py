# ============================================================================
# COMBINED QUIC+TLS — protocol + temporal robustness via SupCon.
# Train = QUIC(W44-45) + TLS(months 1-9). Test = QUIC(W46-47) + TLS(months 10-12).
# Dual eval: (1) temporal generalization early->late, (2) protocol accuracy,
#            (3) temporal INVARIANCE (intra-class cosine across periods).
# ============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, login, HfApi
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
from pathlib import Path
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; torch.backends.cudnn.benchmark=True
HF="donbosoc/shigan-mjkan-baseline"; OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"mjkan_temporal"); Path(OUT).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUT,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
EPOCHS,LR,WARMUP,TAU,BATCH=30,1e-3,3,0.1,512; SEP_W=0.5
CLASSES=["audio_streaming","cdn_web_assets","ecommerce","file_transfer","gaming",
         "messaging","search_info_news","social_media","video_streaming"]
NC=len(CLASSES)

# ---- load both ----
q=np.load(hf_hub_download(HF,"combined_temporal/data/quic_temporal.npz"),allow_pickle=True)
t=np.load(hf_hub_download(HF,"combined_temporal/data/tls_temporal.npz"),allow_pickle=True)
# unify a "period" tag: quic uses week, tls uses month — keep separate masks
q_seq,q_ctx,q_y,q_week=q["seq"],q["ctx"],q["label"].astype(int),q["week"].astype(int)
t_seq,t_ctx,t_y,t_month=t["seq"],t["ctx"],t["label"].astype(int),t["month"].astype(int)

# ---- temporal splits ----
q_train=np.isin(q_week,[44,45]); q_test=np.isin(q_week,[46,47])   # QUIC short horizon
t_train=t_month<=9;              t_test=t_month>=10                # TLS long horizon

# assemble train (union) and keep test sets separate per protocol for eval
TR_seq=np.concatenate([q_seq[q_train],t_seq[t_train]])
TR_ctx=np.concatenate([q_ctx[q_train],t_ctx[t_train]])
TR_y  =np.concatenate([q_y[q_train],  t_y[t_train]])
TR_prot=np.array(["quic"]*q_train.sum()+["tls"]*t_train.sum())
print(f"TRAIN: {len(TR_y)} (quic {q_train.sum()} W44-45 + tls {t_train.sum()} M1-9)")
print(f"TEST quic(W46-47): {q_test.sum()} | TEST tls(M10-12): {t_test.sum()}")
print("train per-class:", {CLASSES[c]:int((TR_y==c).sum()) for c in range(NC)})

# ---- normalization (fit on TRAIN only — honest) ----
m=(np.abs(TR_seq).sum(-1,keepdims=True)>0); fl=TR_seq[m.repeat(5,-1)].reshape(-1,5)
sm,ss=fl.mean(0),fl.std(0)+1e-6
cm,cs=TR_ctx.mean(0),TR_ctx.std(0)+1e-6
def norm_seq(S): mm=(np.abs(S).sum(-1,keepdims=True)>0); return ((S-sm)/ss*mm).astype(np.float32)
def norm_ctx(C): return ((C-cm)/cs).astype(np.float32)
TR_seqn,TR_ctxn=norm_seq(TR_seq),norm_ctx(TR_ctx)

# ---- model (MJKAN deck-faithful) ----
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
        nn.init.zeros_(s.net[-1].weight);nn.init.zeros_(s.net[-1].bias)
    def forward(s,h,c):g=s.net(c);gm,bt=g[:,:s.dm],g[:,s.dm:];gt=s.gate(c);return gt*((1+gm)*h+bt)+(1-gt)*h
class MJKAN(nn.Module):
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__();s.encoder=Encoder(nf,dm);s.film=GatedFiLM(nc,dm)
        s.kan=FasterKANReasoning(dm);s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,x,c): h=s.film(s.encoder(x),c); r=h+s.kan(h); return h,F.normalize(s.proj(r),dim=1)

def supcon(z,y,tau=TAU):
    sim=z@z.T/tau; sim=sim-sim.max(1,keepdim=True).values.detach()
    y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
    lm=torch.ones_like(pos); lm.fill_diagonal_(0)
    exp=torch.exp(sim)*lm; logp=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    base=-((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()
    cents=[F.normalize(z[(y.squeeze()==c)].mean(0),dim=0) for c in y.unique() if (y.squeeze()==c).sum()>0]
    if len(cents)>1: C=torch.stack(cents); inter=(C@C.T).triu(1); sep=inter[inter!=0].mean()
    else: sep=torch.tensor(0.,device=z.device)
    return base+SEP_W*sep

class DS(Dataset):
    def __len__(s): return len(TR_y)
    def __getitem__(s,i): return torch.from_numpy(TR_seqn[i]).float(),torch.from_numpy(TR_ctxn[i]).float(),int(TR_y[i])
cnt=np.bincount(TR_y,minlength=NC); cw=(cnt.sum()/(NC*np.clip(cnt,1,None))).astype(np.float32)
sampler=WeightedRandomSampler(torch.from_numpy(cw[TR_y]).double(),len(TR_y),True)
dl=DataLoader(DS(),batch_size=BATCH,sampler=sampler,num_workers=2,drop_last=True,pin_memory=True)
model=MJKAN(5,TR_ctx.shape[1]).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
print(f"\n{'='*60}\nCOMBINED QUIC+TLS temporal+protocol ({EPOCHS}ep)\n{'='*60}")
for ep in range(EPOCHS):
    for g in opt.param_groups: g["lr"]=LR*(ep+1)/WARMUP if ep<WARMUP else LR
    model.train();tot=0;nb=0
    for seq,ctx,yb in dl:
        seq,ctx,yb=seq.to(DEVICE,non_blocking=True),ctx.to(DEVICE,non_blocking=True),yb.to(DEVICE,non_blocking=True)
        _,z=model(seq,ctx); loss=supcon(z,yb)
        if torch.isnan(loss): continue
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();tot+=loss.item();nb+=1
    if (ep+1)%5==0 or ep==EPOCHS-1: print(f"  ep{ep+1}: loss={tot/nb:.4f}")

# ---- eval helpers ----
@torch.no_grad()
def embed(sq,cx,bs=1024):
    model.eval();Z=[]
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

# reference bank = train embeddings (subsampled, balanced)
ztr=embed(TR_seq,TR_ctx); rng=np.random.default_rng(0)
ri=np.concatenate([rng.choice(np.where(TR_y==c)[0],min(4000,(TR_y==c).sum()),replace=False) for c in range(NC)])
RZ,RY=ztr[ri],TR_y[ri]

# (1) temporal GENERALIZATION early->late, per protocol
zq=embed(q_seq[q_test],q_ctx[q_test]); pq=knn(RZ,RY,zq); acc_q=(pq==q_y[q_test]).mean()
zt=embed(t_seq[t_test],t_ctx[t_test]); pt=knn(RZ,RY,zt); acc_t=(pt==t_y[t_test]).mean()

# (3) temporal INVARIANCE: intra-class cosine, early vs late centroids per protocol
def period_invariance(seq_,ctx_,y_,per_,early_vals,late_vals):
    ze=embed(seq_[np.isin(per_,early_vals)],ctx_[np.isin(per_,early_vals)]); ye=y_[np.isin(per_,early_vals)]
    zl=embed(seq_[np.isin(per_,late_vals)], ctx_[np.isin(per_,late_vals)]);  yl=y_[np.isin(per_,late_vals)]
    cos=[]
    for c in range(NC):
        if (ye==c).sum() and (yl==c).sum():
            ce=ze[ye==c].mean(0); cl=zl[yl==c].mean(0)
            cos.append((ce@cl)/(np.linalg.norm(ce)*np.linalg.norm(cl)+1e-9))
    return float(np.mean(cos))
inv_q=period_invariance(q_seq,q_ctx,q_y,q_week,[44,45],[46,47])
inv_t=period_invariance(t_seq,t_ctx,t_y,t_month,[1,2,3,4,5,6,7,8,9],[10,11,12])

print(f"\n{'='*60}\nCOMBINED — TEMPORAL + PROTOCOL SCORECARD\n{'='*60}")
print(f"  TEMPORAL GENERALIZATION (train early -> test late):")
print(f"    QUIC  W44-45 -> W46-47 (short horizon): {acc_q:.3f}")
print(f"    TLS   M1-9   -> M10-12 (1-year horizon): {acc_t:.3f}")
print(f"  TEMPORAL INVARIANCE (early-vs-late centroid cosine, same class):")
print(f"    QUIC: {inv_q:.3f}   TLS: {inv_t:.3f}   (higher = more time-invariant)")
print(f"  Per-class temporal-generalization (QUIC / TLS):")
for c in range(NC):
    aq=(pq[q_y[q_test]==c]==c).mean() if (q_y[q_test]==c).sum() else float('nan')
    at=(pt[t_y[t_test]==c]==c).mean() if (t_y[t_test]==c).sum() else float('nan')
    print(f"    {CLASSES[c]:18s}: "+(f"{aq:.3f}" if not np.isnan(aq) else "  -  ")+" / "+(f"{at:.3f}" if not np.isnan(at) else "  -  "))

torch.save({"model":model.state_dict(),"norm":{"sm":sm,"ss":ss,"cm":cm,"cs":cs},
            "classes":CLASSES,"acc_quic":float(acc_q),"acc_tls":float(acc_t),
            "inv_quic":float(inv_q),"inv_tls":float(inv_t)},f"{OUT}/combined_temporal_best.pt")
if os.getenv("MJKAN_PUSH_HF") == "1":
    api=HfApi()
    api.upload_file(path_or_fileobj=f"{OUT}/combined_temporal_best.pt",path_in_repo="combined_temporal/combined_temporal_best.pt",repo_id=HF)
    print(f"\n[saved + pushed]")
else:
    print(f"\n[saved locally — use push_to_hf.py mjkan_temporal to upload]")
