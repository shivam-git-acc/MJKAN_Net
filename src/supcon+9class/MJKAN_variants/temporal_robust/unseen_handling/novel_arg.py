# ============================================================================
# LOAD + EVAL unknown_apps_combined on combined_temporal:
#   (1) per-app + overall classification, (2) novelty AUROC + trade-off, (3) gap.
# ============================================================================
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score, roc_curve
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
CLASSES=["audio_streaming","cdn_web_assets","ecommerce","file_transfer","gaming",
         "messaging","search_info_news","social_media","video_streaming"]
NC=9; c2i={c:i for i,c in enumerate(CLASSES)}

# --- MJKAN defs ---
class SSMBlock(nn.Module):
    def __init__(s,dm,ds=16,e=2):
        super().__init__(); s.di=e*dm; s.ds=ds
        s.in_proj=nn.Linear(dm,2*s.di); s.conv=nn.Conv1d(s.di,s.di,3,padding=2,groups=s.di)
        s.x_proj=nn.Linear(s.di,2*ds+1)
        s.A_log=nn.Parameter(torch.log(torch.arange(1,ds+1,dtype=torch.float32).repeat(s.di,1)))
        s.D=nn.Parameter(torch.ones(s.di)); s.out_proj=nn.Linear(s.di,dm)
    def forward(s,x,mask):
        B,L,_=x.shape; xb,z=s.in_proj(x).chunk(2,-1)
        xc=F.silu(s.conv(xb.transpose(1,2))[...,:L].transpose(1,2))
        Bm,Cm,dt=torch.split(s.x_proj(xc),[s.ds,s.ds,1],-1); dt=F.softplus(dt); A=-torch.exp(s.A_log)
        a=torch.exp(dt.unsqueeze(-1)*A.unsqueeze(0).unsqueeze(0)); bx=(dt*Bm).unsqueeze(2)*xc.unsqueeze(-1)
        mm=mask.view(B,L,1,1); a=a*mm+(1-mm); bx=bx*mm
        h=torch.zeros(B,s.di,s.ds,device=x.device); ys=[]
        for tt in range(L): h=a[:,tt]*h+bx[:,tt]; ys.append((h*Cm[:,tt].unsqueeze(1)).sum(-1))
        return s.out_proj((torch.stack(ys,1)+s.D*xc)*F.silu(z))
class Encoder(nn.Module):
    def __init__(s,nf=5,dm=128,nl=2,ds=16,dp=0.1):
        super().__init__(); s.embed=nn.Linear(nf,dm)
        s.blocks=nn.ModuleList([SSMBlock(dm,ds) for _ in range(nl)])
        s.norms=nn.ModuleList([nn.LayerNorm(dm) for _ in range(nl)]); s.drop=nn.Dropout(dp); s.onorm=nn.LayerNorm(dm)
    def forward(s,seq):
        mk=(seq.abs().sum(-1)>0).float(); x=s.embed(seq)
        for b,n in zip(s.blocks,s.norms): x=x+s.drop(b(n(x),mk))
        x=s.onorm(x); mm=mk.unsqueeze(-1); return (x*mm).sum(1)/mm.sum(1).clamp(min=1.0)
class GatedFiLM(nn.Module):
    def __init__(s,n,dm,h=64):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,2*dm))
        s.gate=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,dm),nn.Sigmoid()); s.dm=dm
    def forward(s,h,c): g=s.net(c); gm,bt=g[:,:s.dm],g[:,s.dm:]; gt=s.gate(c); return gt*((1+gm)*h+bt)+(1-gt)*h
class RSWAF(nn.Module):
    def __init__(s,ng=8,gmin=-2.,gmax=2.):
        super().__init__(); g=torch.linspace(gmin,gmax,ng); s.register_buffer("grid",g); s.inv=nn.Parameter(torch.tensor(1.0/((gmax-gmin)/(ng-1))))
    def forward(s,x): return 1.0-torch.tanh((x.unsqueeze(-1)-s.grid)*s.inv)**2
class FasterKANLayer(nn.Module):
    def __init__(s,i,o,ng=8): super().__init__(); s.rswaf=RSWAF(ng); s.ln=nn.LayerNorm(i); s.spline=nn.Linear(i*ng,o); s.base=nn.Linear(i,o)
    def forward(s,x): xn=s.ln(x); return s.spline(s.rswaf(xn).reshape(xn.size(0),-1))+s.base(xn)
class FasterKANReasoning(nn.Module):
    def __init__(s,d,ng=8): super().__init__(); s.l1=FasterKANLayer(d,d,ng); s.bn=nn.BatchNorm1d(d); s.l2=FasterKANLayer(d,d,ng)
    def forward(s,x): return s.l2(F.relu(s.bn(s.l1(x))))
class MJKAN(nn.Module):
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__(); s.encoder=Encoder(nf,dm); s.film=GatedFiLM(nc,dm)
        s.kan=FasterKANReasoning(dm); s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,x,c): h=s.film(s.encoder(x),c); r=h+s.kan(h); return h,F.normalize(s.proj(r),dim=1)

# --- load model + norm ---
ck=torch.load(hf_hub_download(HF,"combined_temporal/combined_temporal_best.pt"),map_location=DEVICE,weights_only=False)
norm=ck["norm"]; sm,ss,cm,cs=[np.array(norm[k],np.float32) for k in ("sm","ss","cm","cs")]; LOG=[0,1]
N_CTX=len(cm); model=MJKAN(5,N_CTX).to(DEVICE); model.load_state_dict(ck["model"]); model.eval()

def norm_ctx_raw(C):  # apply log1p on [0,1] then standardize (RAW ctx -> model input)
    C=C.copy()
    for i in LOG: C[:,i]=np.log1p(np.clip(C[:,i],0,None))
    return ((C-cm)/cs).astype(np.float32)
def norm_seq_raw(S):
    mk=(np.abs(S).sum(-1,keepdims=True)>0); return ((S-sm)/ss*mk).astype(np.float32)
@torch.no_grad()
def embed(seq_n,ctx_n,bs=1024):
    Z=[]
    for i in range(0,len(seq_n),bs):
        Z.append(model(torch.from_numpy(seq_n[i:i+bs]).float().to(DEVICE),
                       torch.from_numpy(ctx_n[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)

# --- centroids from combined_temporal TRAIN ---
q=np.load(hf_hub_download(HF,"combined_temporal/data/quic_temporal.npz"),allow_pickle=True)
t=np.load(hf_hub_download(HF,"combined_temporal/data/tls_temporal.npz"),allow_pickle=True)
ztr=embed(np.concatenate([q["seq"],t["seq"]]),np.concatenate([q["ctx"],t["ctx"]]))  # already normalized
TR_y=np.concatenate([q["label"],t["label"]]).astype(int)
cents=np.stack([ztr[TR_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)

# --- known test confidence ---
qw=q["week"].astype(int); tm=t["month"].astype(int)
zk=embed(np.concatenate([q["seq"][np.isin(qw,[46,47])],t["seq"][tm>=10]]),
         np.concatenate([q["ctx"][np.isin(qw,[46,47])],t["ctx"][tm>=10]]))
zk/=np.linalg.norm(zk,axis=1,keepdims=True); known_conf=(zk@cents.T).max(1)

# --- load unknown apps (RAW) + normalize ---
nv=np.load(hf_hub_download(HF,"generalization/unknown_apps_combined.npz"),allow_pickle=True)
SEQ=norm_seq_raw(nv["seq"]); CTX=norm_ctx_raw(nv["ctx"])
APP=nv["app"].astype(str); PROTO=nv["protocol"].astype(str); EXP=nv["expected_class"].astype(str)
zn=embed(SEQ,CTX); zn/=np.linalg.norm(zn,axis=1,keepdims=True)
sim=zn@cents.T; pred=sim.argmax(1); conf=sim.max(1)

# (1) per-app classification
print("="*78); print("(1) UNSEEN-APP CLASSIFICATION (QUIC+TLS, behavioral expected class)"); print("="*78)
print(f"{'app':22s} {'proto':5s} {'expected':16s} {'landed':16s} {'acc':>6s} {'conf':>6s}")
print("-"*78)
ct=0;nt=0
for a in np.unique(APP):
    m=APP==a; exp=EXP[m][0]; pr=PROTO[m][0]
    landed=CLASSES[np.bincount(pred[m],minlength=9).argmax()]; acc=(pred[m]==c2i[exp]).mean()
    ct+=(pred[m]==c2i[exp]).sum(); nt+=m.sum()
    print(f"{a:22s} {pr:5s} {exp:16s} {landed:16s} {acc:>5.1%} {conf[m].mean():>6.2f}")
print("-"*78); print(f"OVERALL: {ct/nt:.1%} ({nt} flows)")

# (2) novelty detection AUROC + trade-off
print("\n"+"="*78); print("(2) NOVELTY DETECTION — threshold-free"); print("="*78)
labels=np.concatenate([np.ones(len(conf)),np.zeros(len(known_conf))])
scores=np.concatenate([conf,known_conf])
auroc=roc_auc_score(labels,-scores)
print(f"  AUROC (known vs unseen): {auroc:.3f}")
fpr,tpr,_=roc_curve(labels,-scores)
for tf in [0.05,0.10,0.20]:
    i=np.argmin(np.abs(fpr-tf)); print(f"    at {fpr[i]*100:>2.0f}% false alarms: {tpr[i]*100:.0f}% of unseen caught")

# (3) confidence gap
print("\n"+"="*78); print("(3) CONFIDENCE GAP"); print("="*78)
print(f"  known {known_conf.mean():.3f} vs unseen {conf.mean():.3f}  gap {known_conf.mean()-conf.mean():+.3f}")