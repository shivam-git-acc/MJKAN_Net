# ============================================================================
# HIERARCHICAL SupCon + CENTROID SEPARATION (both levels).
# Replaces the loss section of the hierarchical cell — adds separation terms
# that push class centroids apart AND super-group centroids apart.
# ============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from huggingface_hub import hf_hub_download, login, HfApi
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
api=HfApi()
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark=True
HF="donbosoc/shigan-mjkan-baseline"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"hierarchical"); Path(OUT).mkdir(parents=True,exist_ok=True)
import sys as _sys
_lf=open(os.path.join(OUT,"train.log"),"w",buffering=1)
class _Tee:
    def __init__(s,f):s.f=f;s.t=_sys.__stdout__
    def write(s,m):s.t.write(m);s.f.write(m)
    def flush(s):s.t.flush();s.f.flush()
_sys.stdout=_Tee(_lf)
EPOCHS,LR,WARMUP,TAU,BATCH=35,1e-3,2,0.1,512
ALPHA_SG=0.5      # super-group loss weight
BETA_APP=0.1      # gentle app loss weight

meta=json.load(open(hf_hub_download(HF,"protocol_invariance/data/joint_meta.json")))
CATS=meta["categories"]; NC=meta["num_classes"]; N_CTX=meta["n_ctx_feats"]

# data-driven super-groups (k=4 from V7 analysis)
SUPERGROUP_MAP={'audio_streaming':3,'cdn_web_assets':2,'ecommerce':0,'file_transfer':2,
    'gaming':0,'messaging':1,'search_info_news':2,'social_media':1,'video_streaming':2}
SG=np.array([SUPERGROUP_MAP[c] for c in CATS])  # class_idx -> super-group_idx
N_SG=4

dtr=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_train.npz"),allow_pickle=True)
dte=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_test.npz"),allow_pickle=True)
tr_seq,tr_ctx,tr_y=dtr["seq"],dtr["ctx"],dtr["label"].astype(int)
tr_tag=dtr["tag"].astype(str)
te_seq,te_ctx,te_y=dte["seq"],dte["ctx"],dte["label"].astype(int)
te_p=dte["protocol"].astype(str)
# app ids
uapps={a:i for i,a in enumerate(sorted(set(tr_tag)))}
tr_app=np.array([uapps[a] for a in tr_tag])
tr_sg=SG[tr_y]; te_sg=SG[te_y]

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
class Model(nn.Module):
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__();s.encoder=Encoder(nf,dm);s.film=GatedFiLM(nc,dm)
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,x,c):h=s.film(s.encoder(x),c);return h,F.normalize(s.proj(h),dim=1)

def supcon_on(z,labels,tau=TAU):
    sim=z@z.T/tau; sim=sim-sim.max(1,keepdim=True).values.detach()
    labels=labels.view(-1,1); pos=(labels==labels.T).float(); pos.fill_diagonal_(0)
    lm=torch.ones_like(pos); lm.fill_diagonal_(0)
    exp=torch.exp(sim)*lm; logp=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    return -((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()

def centroid_sep(z,labels):
    # push centroids of distinct labels apart (mean pairwise cosine of normalized centroids)
    cents=[]
    for c in labels.unique():
        zc=z[labels==c]
        if len(zc)>0: cents.append(F.normalize(zc.mean(0),dim=0))
    if len(cents)<2: return torch.tensor(0.,device=z.device)
    C=torch.stack(cents)
    cos=C@C.T
    iu=torch.triu_indices(len(C),len(C),1)
    return cos[iu[0],iu[1]].mean()   # lower = more separated; minimizing pushes apart

SG_t=torch.from_numpy(SG).to(DEVICE)
class DS(Dataset):
    def __len__(s): return len(tr_y)
    def __getitem__(s,i): return (torch.from_numpy(tr_seq[i]).float(),torch.from_numpy(tr_ctx[i]).float(),
                                  int(tr_y[i]),int(tr_app[i]))
cnt=np.bincount(tr_y,minlength=NC); cw=(cnt.sum()/(NC*np.clip(cnt,1,None))).astype(np.float32)
sampler=WeightedRandomSampler(torch.from_numpy(cw[tr_y]).double(),len(tr_y),True)
dl=DataLoader(DS(),batch_size=BATCH,sampler=sampler,num_workers=2,drop_last=True,pin_memory=True)

model=Model(meta["n_seq_feats"],N_CTX).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
print(f"{'='*55}\nHIERARCHICAL SupCon (class+supergroup+app)\n{'='*55}")
# weights
ALPHA_SG=0.5      # super-group SupCon
BETA_APP=0.1      # gentle app SupCon
SEP_CLASS=0.5     # class centroid separation (the V7 trick — gives low fine inter-cosine)
SEP_SG=0.3        # super-group centroid separation

print(f"{'='*55}\nHIERARCHICAL SupCon + SEPARATION (both levels)\n{'='*55}")
for ep in range(EPOCHS):
    for g in opt.param_groups: g["lr"]=LR*(ep+1)/WARMUP if ep<WARMUP else LR
    model.train();tc=ts=ta=tsc=tss=0;nb=0
    for seq,ctx,yb,appb in dl:
        seq,ctx,yb,appb=[x.to(DEVICE,non_blocking=True) for x in (seq,ctx,yb,appb)]
        sgb=SG_t[yb]
        _,z=model(seq,ctx)
        lc=supcon_on(z,yb)                # class clustering
        lsg=supcon_on(z,sgb)              # super-group clustering
        la=supcon_on(z,appb)             # gentle app clustering
        sep_c=centroid_sep(z,yb)          # push class centroids apart
        sep_sg=centroid_sep(z,sgb)        # push super-group centroids apart
        loss = lc + ALPHA_SG*lsg + BETA_APP*la + SEP_CLASS*sep_c + SEP_SG*sep_sg
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        tc+=lc.item();ts+=lsg.item();ta+=la.item();tsc+=sep_c.item();tss+=sep_sg.item();nb+=1
    if (ep+1)%5==0 or ep==EPOCHS-1:
        print(f"  ep{ep+1}: class={tc/nb:.3f} sg={ts/nb:.3f} app={ta/nb:.3f} sep_c={tsc/nb:.3f} sep_sg={tss/nb:.3f}")

# ===== EVAL =====
@torch.no_grad()
def embed(sq,cx,bs=1024):
    model.eval();Z=[]
    for i in range(0,len(sq),bs):
        Z.append(model(torch.from_numpy(sq[i:i+bs]).float().to(DEVICE),torch.from_numpy(cx[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,nc,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T;idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=nc).argmax() for v in idx]))
    return np.concatenate(out)
ztr=embed(tr_seq,tr_ctx); zte=embed(te_seq,te_ctx)
rng=np.random.default_rng(0)
ri=np.concatenate([rng.choice(np.where(tr_y==c)[0],min(4000,(tr_y==c).sum()),replace=False) for c in range(NC)])

# fine-class accuracy
pred_c=knn(ztr[ri],tr_y[ri],zte,NC)
qa=(pred_c[te_p=="quic"]==te_y[te_p=="quic"]).mean(); ta_=(pred_c[te_p=="tls"]==te_y[te_p=="tls"]).mean()
# super-group accuracy (predict super-group directly)
pred_sg=knn(ztr[ri],tr_sg[ri],zte,N_SG)
sg_acc=(pred_sg==te_sg).mean()
# fine-class centroids cosine (embedding KPI)
cents=np.stack([ztr[tr_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
inter=(cents@cents.T)[np.triu_indices(NC,1)].mean()
# super-group centroid separation
sgc=np.stack([ztr[tr_sg==g].mean(0) for g in range(N_SG)]); sgc/=np.linalg.norm(sgc,axis=1,keepdims=True)
sg_inter=(sgc@sgc.T)[np.triu_indices(N_SG,1)].mean()

print(f"\n{'='*55}\nHIERARCHICAL MODEL RESULTS\n{'='*55}")
print(f"  Fine-class accuracy:  QUIC={qa:.3f} TLS={ta_:.3f}")
print(f"  SUPER-GROUP accuracy: {sg_acc:.3f}  (known test set)")
print(f"  Fine inter-cosine:    {inter:.3f}")
print(f"  Super-group inter-cosine: {sg_inter:.3f} (lower=better separated)")
print(f"  Super-group per-group acc:")
for g in range(N_SG):
    m=te_sg==g; members=[CATS[i] for i in range(NC) if SG[i]==g]
    print(f"    G{g} {str(members)[:40]:40s}: {(pred_sg[m]==g).mean():.3f}")
torch.save({"model":model.state_dict(),"meta":meta,"supergroup_map":SUPERGROUP_MAP,
            "fine_quic":float(qa),"fine_tls":float(ta_),"sg_acc":float(sg_acc)},f"{OUT}/hierarchical_best.pt")
print("\n[saved hierarchical model]")