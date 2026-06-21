# ============================================================================
# V2 — FiLM CONDITIONING (from scratch). Encoder -> FiLM(h | 6-feat context)
# -> projection -> z, SupCon. Tests whether conditioning on behavioral context
# improves invariance over V1. FiLM uses the 6 ctx features ONLY (no protocol).
# Same recipe as V1. Eval QUIC/TLS separately.
# ============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from huggingface_hub import hf_hub_download, login
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
OUTDIR=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"v2"); Path(OUTDIR).mkdir(parents=True,exist_ok=True)
BATCH, EPOCHS, LR, WARMUP, TAU, SEED = 512, 50, 1e-3, 2, 0.1, 0
torch.manual_seed(SEED); np.random.seed(SEED)

meta=json.load(open(hf_hub_download(HF,"protocol_invariance/data/joint_meta.json")))
CATS=meta["categories"]; NC=meta["num_classes"]; CW=np.array(meta["class_weights"],np.float32)
N_CTX=meta["n_ctx_feats"]
print(f"[joint] {NC} classes, {N_CTX} context features for FiLM")

class DS(Dataset):
    def __init__(s,sp):
        d=np.load(hf_hub_download(HF,f"protocol_invariance/data/joint_{sp}.npz"),allow_pickle=True)
        s.seq=d["seq"]; s.ctx=d["ctx"]; s.y=d["label"].astype(int); s.proto=d["protocol"].astype(str)
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return (torch.from_numpy(s.seq[i]).float(),
                                  torch.from_numpy(s.ctx[i]).float(), int(s.y[i]))
tr=DS("train"); te=DS("test")
w=CW[tr.y]; sampler=WeightedRandomSampler(torch.from_numpy(w).double(),len(w),True)
tl=DataLoader(tr,batch_size=BATCH,sampler=sampler,num_workers=2,drop_last=True)

# ---- model: Encoder (verbatim) + FiLM + projection ----
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

class FiLM(nn.Module):
    """Context (6 feats) -> gamma, beta (D_MODEL each) -> modulate h."""
    def __init__(s, n_ctx, dm, hidden=64):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(n_ctx,hidden), nn.ReLU(), nn.Linear(hidden,2*dm))
        s.dm=dm
        # init last layer so FiLM starts near identity (gamma~1, beta~0)
        nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)
    def forward(s, h, ctx):
        gb=s.net(ctx); gamma,beta=gb[:,:s.dm],gb[:,s.dm:]
        return (1.0+gamma)*h + beta          # gamma offset by 1 -> identity at init

class FiLMSupConModel(nn.Module):
    def __init__(s,nf=5,n_ctx=6,dm=128,nl=2,ds=16,dp=0.1):
        super().__init__()
        s.encoder=Encoder(nf,dm,nl,ds,dp)
        s.film=FiLM(n_ctx,dm)
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,seq,ctx):
        h=s.encoder(seq)
        h=s.film(h,ctx)                       # <<< FiLM conditioning on context
        z=F.normalize(s.proj(h),dim=1)
        return h,z

def supcon_loss(z,y,tau=TAU):
    z=F.normalize(z,dim=1); sim=z@z.T/tau
    sim=sim-sim.max(1,keepdim=True).values.detach()
    y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
    lm=torch.ones_like(pos); lm.fill_diagonal_(0)
    exp=torch.exp(sim)*lm
    logp=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    mpp=(pos*logp).sum(1)/pos.sum(1).clamp(min=1)
    return -mpp.mean()

model=FiLMSupConModel(meta["n_seq_feats"],N_CTX,128,2,16,0.1).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
def lr_at(ep): return LR*(ep+1)/WARMUP if ep<WARMUP else LR

@torch.no_grad()
def embed_z(ds,bs=512):
    model.eval(); Z=[]
    for i in range(0,len(ds.seq),bs):
        sb=torch.from_numpy(ds.seq[i:i+bs]).float().to(DEVICE)
        cb=torch.from_numpy(ds.ctx[i:i+bs]).float().to(DEVICE)
        Z.append(model(sb,cb)[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T; idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)
def evalp(tag):
    ztr=embed_z(tr); zte=embed_z(te)
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

best=0
for ep in range(EPOCHS):
    model.train()
    for g in opt.param_groups: g["lr"]=lr_at(ep)
    tot=0; nb=0
    for seq,ctx,y in tl:
        seq,ctx,y=seq.to(DEVICE),ctx.to(DEVICE),y.to(DEVICE)
        _,z=model(seq,ctx); loss=supcon_loss(z,y)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    if (ep+1)%5==0 or ep==EPOCHS-1:
        ov,qa,ta=evalp(f"ep{ep+1}")
        print(f"[ep{ep+1}] loss={tot/nb:.4f} overall={ov:.3f} TLS={ta:.3f}")
        if ta>best:
            best=ta; torch.save({"model":model.state_dict(),"meta":meta,"epoch":ep+1,
                "tls_acc":ta,"quic_acc":qa,"overall":ov},f"{OUTDIR}/v2_film_best.pt")
            print(f"   ** saved (TLS={ta:.3f}, QUIC={qa:.3f})")
    else:
        print(f"[ep{ep+1}] loss={tot/nb:.4f}")
print(f"\nV2 FiLM DONE. Best TLS={best:.3f} (V1=0.72, V5z=0.734). Did context-conditioning help?")