import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, math
from pathlib import Path
from huggingface_hub import hf_hub_download
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"transformer"); Path(OUT).mkdir(parents=True,exist_ok=True)

if os.getenv("MJKAN_FROM_HF") == "1":
    CKPT=hf_hub_download(HF,"transformer/transformer_best.pt")
else:
    CKPT=str(Path(OUT,"transformer_best.pt"))

# NOTE: the train script (transformer_train) does NOT call torch.save().
# To use this eval script after training, add to the end of transformer_train:
#   torch.save({"model": model.state_dict(), "meta": meta}, f"{OUT}/transformer_best.pt")

class PosEnc(nn.Module):
    def __init__(s,dm,L=30):
        super().__init__(); pe=torch.zeros(L,dm)
        pos=torch.arange(L).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dm,2).float()*(-math.log(10000)/dm))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        s.register_buffer("pe",pe.unsqueeze(0))
    def forward(s,x): return x+s.pe[:,:x.size(1)]
class TransformerEnc(nn.Module):
    def __init__(s,nf=5,dm=128,nhead=4,nl=3,dp=0.1):
        super().__init__(); s.embed=nn.Linear(nf,dm); s.pos=PosEnc(dm)
        layer=nn.TransformerEncoderLayer(dm,nhead,dim_feedforward=4*dm,dropout=dp,batch_first=True,activation="gelu")
        s.tf=nn.TransformerEncoder(layer,nl); s.onorm=nn.LayerNorm(dm)
    def forward(s,seq):
        mk=(seq.abs().sum(-1)>0)
        x=s.pos(s.embed(seq)); x=s.tf(x,src_key_padding_mask=~mk)
        mm=mk.unsqueeze(-1).float()
        return s.onorm((x*mm).sum(1)/mm.sum(1).clamp(min=1.0))
class GatedFiLM(nn.Module):
    def __init__(s,n,dm,h=64):
        super().__init__();s.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,2*dm))
        s.gate=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,dm),nn.Sigmoid());s.dm=dm
        nn.init.zeros_(s.net[-1].weight);nn.init.zeros_(s.net[-1].bias)
    def forward(s,h,c):g=s.net(c);gm,bt=g[:,:s.dm],g[:,s.dm:];gt=s.gate(c);return gt*((1+gm)*h+bt)+(1-gt)*h
class Model(nn.Module):
    def __init__(s,nf=5,nc=6,dm=128):
        super().__init__();s.encoder=TransformerEnc(nf,dm);s.film=GatedFiLM(nc,dm)
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,x,c):h=s.film(s.encoder(x),c);return F.normalize(s.proj(h),dim=1)

ck=torch.load(CKPT,map_location=DEVICE,weights_only=False)
meta=ck["meta"]; CATS=meta["categories"]; NC=meta["num_classes"]; N_CTX=meta["n_ctx_feats"]
model=Model(meta["n_seq_feats"],N_CTX).to(DEVICE)
model.load_state_dict(ck["model"]); model.eval()
print(f"[TRANSFORMER] loaded | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

dtr=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_train.npz"),allow_pickle=True)
dte=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_test.npz"),allow_pickle=True)
tr_seq,tr_ctx,tr_y=dtr["seq"],dtr["ctx"],dtr["label"].astype(int)
te_seq,te_ctx,te_y,te_p=dte["seq"],dte["ctx"],dte["label"].astype(int),dte["protocol"].astype(str)

@torch.no_grad()
def embed(sq,cx,bs=512):
    model.eval();Z=[]
    for i in range(0,len(sq),bs):
        # Model forward returns z directly (not a tuple)
        Z.append(model(torch.from_numpy(sq[i:i+bs]).float().to(DEVICE),
                       torch.from_numpy(cx[i:i+bs]).float().to(DEVICE)).cpu().numpy())
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
ts=[]
for _ in range(200):
    a=time.perf_counter()
    with torch.no_grad(): _=model(x1,c1)
    ts.append((time.perf_counter()-a)*1000)

print(f"\n{'='*55}\nTRANSFORMER — EVAL SCORECARD\n{'='*55}")
print(f"  Overall:  {ov:.3f} | QUIC: {qa:.3f} | TLS: {ta:.3f}")
print(f"  Intra: {intra:.3f} (target >0.7) | Inter: {inter:.3f} (target <0.3)")
print(f"  Latency: {np.median(ts):.2f}ms")
print(f"  vs SSM baseline: QUIC 0.877 / TLS 0.780")
print(f"\n  Per-class accuracy:")
for c in range(NC):
    m=te_y==c; print(f"    {CATS[c]:18s}: {(pred[m]==c).mean():.3f}")

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
    plt.title(f"TRANSFORMER — QUIC ○ / TLS △ | TLS={ta:.3f} inter={inter:.3f}")
    plt.legend(fontsize=7,ncol=2); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(str(Path(OUT,"transformer_tsne.png")),dpi=130,bbox_inches="tight")
    print(f"[saved] {OUT}/transformer_tsne.png")
except Exception as e: print("[tsne skipped]",e)

result={"model":"transformer","overall":float(ov),"quic":float(qa),"tls":float(ta),
        "intra":float(intra),"inter":float(inter),"latency_ms":float(np.median(ts)),
        "per_class":{CATS[c]:float((pred[te_y==c]==c).mean()) for c in range(NC)}}
Path(f"{OUT}/transformer_eval.json").write_text(json.dumps(result,indent=2))
print(f"[saved] {OUT}/transformer_eval.json")
