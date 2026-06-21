import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from pathlib import Path
from huggingface_hub import hf_hub_download
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"; HF_DATA="protocol_invariance/data"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"v5_dann"); Path(OUT).mkdir(parents=True,exist_ok=True)

if os.getenv("MJKAN_FROM_HF") == "1":
    CKPT=hf_hub_download(HF,"protocol_invariance/v5z_dann/supcon_v5z_best.pt")
else:
    CKPT=str(Path(OUT,"v5z_dann_best.pt"))

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

ck=torch.load(CKPT,map_location=DEVICE,weights_only=False)
meta=ck["meta"]; CATS=meta["categories"]; NC=meta["num_classes"]
model=SupConModel(meta["n_seq_feats"],128,2,16,0.1).to(DEVICE)
model.load_state_dict(ck["model"]); model.eval()
print(f"[V5z-DANN] loaded | TLS={ck.get('tls_acc',0):.3f} QUIC={ck.get('quic_acc',0):.3f} dom_acc={ck.get('dom_acc',0):.3f}")

dtr=np.load(hf_hub_download(HF,f"{HF_DATA}/joint_train.npz"),allow_pickle=True)
dte=np.load(hf_hub_download(HF,f"{HF_DATA}/joint_test.npz"),allow_pickle=True)
tr_seq,tr_y=dtr["seq"],dtr["label"].astype(int)
te_seq,te_y,te_p=dte["seq"],dte["label"].astype(int),dte["protocol"].astype(str)

@torch.no_grad()
def embed(seq,bs=512):
    model.eval();Z=[]
    for i in range(0,len(seq),bs):
        Z.append(model(torch.from_numpy(seq[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T;idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)

ztr=embed(tr_seq); zte=embed(te_seq)
rng=np.random.default_rng(0)
ri=np.concatenate([rng.choice(np.where(tr_y==c)[0],min(2000,(tr_y==c).sum()),replace=False) for c in range(NC)])
pred=knn(ztr[ri],tr_y[ri],zte)
q=te_p=="quic"; t=te_p=="tls"
ov=(pred==te_y).mean(); qa=(pred[q]==te_y[q]).mean(); ta=(pred[t]==te_y[t]).mean()
cents=np.stack([zte[te_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
intra=float(np.mean([(zte[te_y==c]@cents[c]).mean() for c in range(NC)]))
inter=float((cents@cents.T)[np.triu_indices(NC,1)].mean())

x1=torch.from_numpy(te_seq[:1]).float().to(DEVICE)
for _ in range(20):
    with torch.no_grad(): _=model(x1)
ts=[]
for _ in range(200):
    a=time.perf_counter()
    with torch.no_grad(): _=model(x1)
    ts.append((time.perf_counter()-a)*1000)

print(f"\n{'='*55}\nV5z-DANN — EVAL SCORECARD\n{'='*55}")
print(f"  Overall:  {ov:.3f} | QUIC: {qa:.3f} | TLS: {ta:.3f}")
print(f"  Intra: {intra:.3f} (target >0.7) | Inter: {inter:.3f} (target <0.3)")
print(f"  Latency: {np.median(ts):.2f}ms")
print(f"\n  Per-class TLS accuracy:")
for c in range(NC):
    m=t&(te_y==c)
    if m.sum(): print(f"    {CATS[c]:18s} {(pred[m]==c).mean():.3f}")
print(f"\n  Comparison: V1={0.720:.3f} V5h={0.727:.3f} (TLS)")

try:
    from sklearn.manifold import TSNE; import matplotlib.pyplot as plt
    ids=np.concatenate([rng.choice(np.where(te_y==c)[0],min(250,(te_y==c).sum()),replace=False) for c in range(NC)])
    emb=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(zte[ids])
    yy=te_y[ids]; pp=te_p[ids]; cmap=plt.cm.tab10
    plt.figure(figsize=(10,8))
    for c in range(NC):
        mq=(yy==c)&(pp=="quic"); mt=(yy==c)&(pp=="tls")
        if mq.sum(): plt.scatter(emb[mq,0],emb[mq,1],s=12,marker="o",alpha=0.5,color=cmap(c%10),label=CATS[c])
        if mt.sum(): plt.scatter(emb[mt,0],emb[mt,1],s=24,marker="^",alpha=0.8,edgecolors="k",linewidths=0.3,color=cmap(c%10))
    plt.title(f"V5z-DANN — QUIC ○ / TLS △ | TLS={ta:.3f} intra={intra:.3f} inter={inter:.3f}")
    plt.legend(fontsize=7,ncol=2); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(str(Path(OUT,"v5_tsne.png")),dpi=130,bbox_inches="tight")
    print(f"[saved] {OUT}/v5_tsne.png")
except Exception as e: print("[tsne skipped]",e)

result={"model":"V5z-DANN","overall":float(ov),"quic":float(qa),"tls":float(ta),
        "intra":float(intra),"inter":float(inter),"latency_ms":float(np.median(ts)),
        "per_class_tls":{CATS[c]:float((pred[t&(te_y==c)]==c).mean()) for c in range(NC) if (t&(te_y==c)).sum()}}
Path(f"{OUT}/v5_eval.json").write_text(json.dumps(result,indent=2))
print(f"[saved] {OUT}/v5_eval.json")
