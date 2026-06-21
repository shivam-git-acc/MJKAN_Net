import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from pathlib import Path
from huggingface_hub import hf_hub_download
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"mjkan"); Path(OUT).mkdir(parents=True,exist_ok=True)

if os.getenv("MJKAN_FROM_HF") == "1":
    CKPT=hf_hub_download(HF,"mjkan/mjkan_best.pt")
else:
    CKPT=str(Path(OUT,"mjkan_best.pt"))

class RSWAF(nn.Module):
    def __init__(s,num_grids=8,gmin=-2.0,gmax=2.0):
        super().__init__(); g=torch.linspace(gmin,gmax,num_grids)
        s.register_buffer("grid",g); s.inv=nn.Parameter(torch.tensor(1.0/((gmax-gmin)/(num_grids-1))))
    def forward(s,x): return 1.0-torch.tanh((x.unsqueeze(-1)-s.grid)*s.inv)**2
class FasterKANLayer(nn.Module):
    def __init__(s,i,o,num_grids=8):
        super().__init__();s.rswaf=RSWAF(num_grids);s.ln=nn.LayerNorm(i)
        s.spline=nn.Linear(i*num_grids,o);s.base=nn.Linear(i,o)
    def forward(s,x): xn=s.ln(x); b=s.rswaf(xn).reshape(xn.size(0),-1); return s.spline(b)+s.base(xn)
class FasterKANReasoning(nn.Module):
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
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__();s.encoder=Encoder(nf,dm);s.film=GatedFiLM(nc,dm)
        s.kan=FasterKANReasoning(dm)
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,x,c):
        h=s.film(s.encoder(x),c); r=s.kan(h); r=h+r; z=s.proj(r); return h,F.normalize(z,dim=1)

ck=torch.load(CKPT,map_location=DEVICE,weights_only=False)
meta=ck["meta"]; CATS=meta["categories"]; NC=meta["num_classes"]; N_CTX=meta["n_ctx_feats"]
model=MJKAN(meta["n_seq_feats"],N_CTX).to(DEVICE)
model.load_state_dict(ck["model"]); model.eval()
print(f"[MJKAN] loaded | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"  stored: QUIC={ck.get('acc_quic',0):.3f} TLS={ck.get('acc_tls',0):.3f}")

dtr=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_train.npz"),allow_pickle=True)
dte=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_test.npz"),allow_pickle=True)
tr_seq,tr_ctx,tr_y=dtr["seq"],dtr["ctx"],dtr["label"].astype(int)
te_seq,te_ctx,te_y,te_p=dte["seq"],dte["ctx"],dte["label"].astype(int),dte["protocol"].astype(str)

@torch.no_grad()
def embed(sq,cx,bs=512):
    model.eval();Z=[]
    for i in range(0,len(sq),bs):
        Z.append(model(torch.from_numpy(sq[i:i+bs]).float().to(DEVICE),
                       torch.from_numpy(cx[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T;idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)

ztr=embed(tr_seq,tr_ctx); zte=embed(te_seq,te_ctx)
rng=np.random.default_rng(0)
ri=np.concatenate([rng.choice(np.where(tr_y==c)[0],min(4000,(tr_y==c).sum()),replace=False) for c in range(NC)])
pred=knn(ztr[ri],tr_y[ri],zte)
q=te_p=="quic"; t=te_p=="tls"
ov=(pred==te_y).mean(); qa=(pred[q]==te_y[q]).mean(); ta=(pred[t]==te_y[t]).mean()
cents=np.stack([ztr[tr_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
intra=float(np.mean([(zte[te_y==c]@cents[c]).mean() for c in range(NC)]))
inter=float((cents@cents.T)[np.triu_indices(NC,1)].mean())

x1=torch.from_numpy(te_seq[:1]).float().to(DEVICE); c1=torch.from_numpy(te_ctx[:1]).float().to(DEVICE)
for _ in range(20):
    with torch.no_grad(): _=model(x1,c1)
if DEVICE=="cuda": torch.cuda.synchronize()
ts=[]
for _ in range(200):
    a=time.perf_counter()
    with torch.no_grad(): _=model(x1,c1)
    if DEVICE=="cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter()-a)*1000)

print(f"\n{'='*55}\nMJKAN — KPI SCORECARD\n{'='*55}")
print(f"  Overall:  {ov:.3f} | QUIC: {qa:.3f} | TLS: {ta:.3f}")
print(f"  Intra: {intra:.3f} (target >0.7) | Inter: {inter:.3f} (target <0.3) {'✓' if inter<0.3 else ''}")
print(f"  Latency: {np.median(ts):.2f}ms (target <5ms)")
print(f"  vs V7 (SSM+MLP): QUIC 0.877 / TLS 0.737 / inter 0.234")
print(f"\n  Per-class accuracy:")
for c in range(NC):
    mq=(te_y==c)&q; mt=(te_y==c)&t
    aq=(pred[mq]==c).mean() if mq.sum() else float('nan')
    at=(pred[mt]==c).mean() if mt.sum() else float('nan')
    print(f"    {CATS[c]:18s}: QUIC={aq:.3f}  TLS={'(none)' if mt.sum()==0 else f'{at:.3f}'}")

try:
    from sklearn.manifold import TSNE; import matplotlib.pyplot as plt
    ids=np.concatenate([rng.choice(np.where(te_y==c)[0],min(250,(te_y==c).sum()),replace=False) for c in range(NC)])
    emb=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(zte[ids])
    yy=te_y[ids]; pp2=te_p[ids]; cmap=plt.cm.tab10
    plt.figure(figsize=(10,8))
    for c in range(NC):
        mq=(yy==c)&(pp2=="quic"); mt=(yy==c)&(pp2=="tls")
        if mq.sum(): plt.scatter(emb[mq,0],emb[mq,1],s=12,marker="o",alpha=0.5,color=cmap(c%10),label=CATS[c])
        if mt.sum(): plt.scatter(emb[mt,0],emb[mt,1],s=24,marker="^",alpha=0.8,edgecolors="k",linewidths=0.3,color=cmap(c%10))
    plt.title(f"MJKAN — QUIC ○ / TLS △ | TLS={ta:.3f} intra={intra:.3f} inter={inter:.3f}")
    plt.legend(fontsize=7,ncol=2); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(str(Path(OUT,"mjkan_tsne.png")),dpi=130,bbox_inches="tight")
    print(f"[saved] {OUT}/mjkan_tsne.png")
except Exception as e: print("[tsne skipped]",e)

result={"model":"MJKAN","overall":float(ov),"quic":float(qa),"tls":float(ta),
        "intra":float(intra),"inter":float(inter),"latency_ms":float(np.median(ts)),
        "inter_kpi_pass":bool(inter<0.3),"intra_kpi_pass":bool(intra>0.7),"latency_kpi_pass":bool(np.median(ts)<5.0),
        "per_class":{CATS[c]:{"quic":float((pred[(te_y==c)&q]==c).mean()) if ((te_y==c)&q).sum() else None,
                               "tls":float((pred[(te_y==c)&t]==c).mean()) if ((te_y==c)&t).sum() else None} for c in range(NC)}}
Path(f"{OUT}/mjkan_eval.json").write_text(json.dumps(result,indent=2))
print(f"[saved] {OUT}/mjkan_eval.json")
