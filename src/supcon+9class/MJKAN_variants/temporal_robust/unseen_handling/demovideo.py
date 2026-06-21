#!/usr/bin/env python3
# =============================================================================
# STEP 1 (run on Kaggle): export demo data to demo_data.json for the HTML viewer.
# Runs the flagship on the published unknown-app benchmark, dumps per-app:
#   one example flow (sizes+dirs), class distribution, expected class,
#   confidence, verdict. The HTML viewer reads this JSON — no PyTorch in browser.
# =============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download, login
if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"]=UserSecretsClient().get_secret("HF_TOKEN")
    except Exception: pass
if os.environ.get("HF_TOKEN"): login(token=os.environ["HF_TOKEN"])
HF="donbosoc/shigan-mjkan-baseline"; DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CLASSES=["audio_streaming","cdn_web_assets","ecommerce","file_transfer","gaming",
         "messaging","search_info_news","social_media","video_streaming"]; NC=9; LOG=[0,1]

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

ck=torch.load(hf_hub_download(HF,"combined_temporal/combined_temporal_best.pt"),map_location=DEVICE,weights_only=False)
sm,ss,cm,cs=[np.array(ck["norm"][k],np.float32) for k in ("sm","ss","cm","cs")]; n_ctx=len(cm)
model=MJKAN(5,n_ctx).to(DEVICE); model.load_state_dict(ck["model"],strict=True); model.eval()

@torch.no_grad()
def embed(seq_raw,ctx_raw,bs=1024):
    Z=[]
    for i in range(0,len(seq_raw),bs):
        S=seq_raw[i:i+bs].astype(np.float32); mk=(np.abs(S).sum(-1,keepdims=True)>0)
        Sn=((S-sm)/ss*mk).astype(np.float32); C=ctx_raw[i:i+bs].astype(np.float32).copy()
        for j in LOG: C[:,j]=np.log1p(np.clip(C[:,j],0,None))
        Cn=((C-cm)/cs).astype(np.float32)
        Z.append(model(torch.from_numpy(Sn).float().to(DEVICE),torch.from_numpy(Cn).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)

# centroids + known confidence
q=np.load(hf_hub_download(HF,"combined_temporal/data/quic_temporal.npz"),allow_pickle=True)
t=np.load(hf_hub_download(HF,"combined_temporal/data/tls_temporal.npz"),allow_pickle=True)
qw=q["week"].astype(int); tm=t["month"].astype(int)
Ztr=embed(np.concatenate([q["seq"][np.isin(qw,[44,45])],t["seq"][tm<=9]]),
          np.concatenate([q["ctx"][np.isin(qw,[44,45])],t["ctx"][tm<=9]]))
TRy=np.concatenate([q["label"][np.isin(qw,[44,45])],t["label"][tm<=9]]).astype(int)
cents=np.zeros((NC,Ztr.shape[1]),np.float32)
for c in range(NC):
    if (TRy==c).sum(): cents[c]=Ztr[TRy==c].mean(0)
cents/=np.clip(np.linalg.norm(cents,axis=1,keepdims=True),1e-9,None)
Zk=embed(np.concatenate([q["seq"][np.isin(qw,[46,47])],t["seq"][tm>=10]]),
         np.concatenate([q["ctx"][np.isin(qw,[46,47])],t["ctx"][tm>=10]]))
Zk/=np.linalg.norm(Zk,axis=1,keepdims=True); known_mean=float((Zk@cents.T).max(1).mean())

# unknown apps
nv=np.load(hf_hub_download(HF,"generalization/unknown_apps_combined.npz"),allow_pickle=True)
U_seq=nv["seq"].astype(np.float32); U_ctx=nv["ctx"].astype(np.float32)
U_app=nv["app"].astype(str); U_exp=nv["expected_class"].astype(str)
U_proto=nv["protocol"].astype(str) if "protocol" in nv else np.array(["?"]*len(U_app))
Zu=embed(U_seq,U_ctx); Zu/=np.linalg.norm(Zu,axis=1,keepdims=True)
sim=Zu@cents.T; pred=sim.argmax(1); conf=sim.max(1)

# story order: strong generalisers first (distinctive behaviour), then
# predictable-overlap + novelty-detection cases. Edit freely.
ORDER=["bing","duckduckgo","seznam-search","apple-itunes","ctu-matrix","slack",
       "overleaf-cdn","soundcloud","steam","xbox-live","microsoft-onedrive","apple-icloud"]
present=set(U_app)
apps=[a for a in ORDER if a in present]
apps += [a for a in sorted(present) if a not in apps]   # top up with the rest
apps = apps[:12]   # cap for a tight video

out={"classes":[c.replace("_"," ") for c in CLASSES],"known_mean":round(known_mean,3),"apps":[]}
for a in apps:
    m=U_app==a; exp=U_exp[m][0]
    ex=U_seq[m][0]; valid=np.abs(ex).sum(-1)>0
    flow=[{"size":float(abs(ex[i,0])),"dir":int(np.sign(ex[i,1])),"iat":float(ex[i,2])} for i in range(int(valid.sum()))]
    dist=(np.bincount(pred[m],minlength=NC)/m.sum()).round(3).tolist()
    top=int(np.argmax(dist)); acc=float((pred[m]==CLASSES.index(exp)).mean()) if exp in CLASSES else None
    cf=float(conf[m].mean())
    out["apps"].append({
        "app":a,"protocol":U_proto[m][0],"expected":exp.replace("_"," "),
        "expected_idx":CLASSES.index(exp) if exp in CLASSES else -1,
        "flow":flow,"dist":dist,"top":top,"top_class":CLASSES[top].replace("_"," "),
        "acc":(round(acc,3) if acc is not None else None),
        "conf":round(cf,3),"n":int(m.sum()),
        "success":bool(top==CLASSES.index(exp)) if exp in CLASSES else False,
        "flagged":bool(cf<known_mean-0.05),
    })
json.dump(out,open("demo_data.json","w"),indent=1)
print(f"[saved] demo_data.json  ({len(out['apps'])} apps, known_mean={known_mean:.3f})")
print("Apps:", [a['app'] for a in out['apps']])