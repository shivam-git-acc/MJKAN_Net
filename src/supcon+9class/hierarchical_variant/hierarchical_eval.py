import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from pathlib import Path
from huggingface_hub import hf_hub_download
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
OUT=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"hierarchical"); Path(OUT).mkdir(parents=True,exist_ok=True)

if os.getenv("MJKAN_FROM_HF") == "1":
    CKPT=hf_hub_download(HF,"hierarchical/hierarchical_best.pt")
else:
    CKPT=str(Path(OUT,"hierarchical_best.pt"))

SUPERGROUP_MAP={'audio_streaming':3,'cdn_web_assets':2,'ecommerce':0,'file_transfer':2,
    'gaming':0,'messaging':1,'search_info_news':2,'social_media':1,'video_streaming':2}
N_SG=4

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

ck=torch.load(CKPT,map_location=DEVICE,weights_only=False)
meta=ck["meta"]; CATS=meta["categories"]; NC=meta["num_classes"]; N_CTX=meta["n_ctx_feats"]
SG=np.array([SUPERGROUP_MAP[c] for c in CATS])
model=Model(meta["n_seq_feats"],N_CTX).to(DEVICE)
model.load_state_dict(ck["model"]); model.eval()
print(f"[HIER] loaded | fine_quic={ck.get('fine_quic',0):.3f} fine_tls={ck.get('fine_tls',0):.3f} sg_acc={ck.get('sg_acc',0):.3f}")

dtr=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_train.npz"),allow_pickle=True)
dte=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_test.npz"),allow_pickle=True)
tr_seq,tr_ctx,tr_y=dtr["seq"],dtr["ctx"],dtr["label"].astype(int)
te_seq,te_ctx,te_y,te_p=dte["seq"],dte["ctx"],dte["label"].astype(int),dte["protocol"].astype(str)
tr_sg=SG[tr_y]; te_sg=SG[te_y]

@torch.no_grad()
def embed(sq,cx,bs=1024):
    model.eval();Z=[]
    for i in range(0,len(sq),bs):
        Z.append(model(torch.from_numpy(sq[i:i+bs]).float().to(DEVICE),
                       torch.from_numpy(cx[i:i+bs]).float().to(DEVICE))[1].cpu().numpy())
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

pred_c=knn(ztr[ri],tr_y[ri],zte,NC)
q=te_p=="quic"; t=te_p=="tls"
qa=(pred_c[q]==te_y[q]).mean(); ta=(pred_c[t]==te_y[t]).mean(); ov=(pred_c==te_y).mean()

pred_sg=knn(ztr[ri],tr_sg[ri],zte,N_SG); sg_acc=(pred_sg==te_sg).mean()

cents=np.stack([ztr[tr_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
fine_inter=float((cents@cents.T)[np.triu_indices(NC,1)].mean())
fine_intra=float(np.mean([(zte[te_y==c]@cents[c]).mean() for c in range(NC)]))
sgc=np.stack([ztr[tr_sg==g].mean(0) for g in range(N_SG)]); sgc/=np.linalg.norm(sgc,axis=1,keepdims=True)
sg_inter=float((sgc@sgc.T)[np.triu_indices(N_SG,1)].mean())

x1=torch.from_numpy(te_seq[:1]).float().to(DEVICE); c1=torch.from_numpy(te_ctx[:1]).float().to(DEVICE)
for _ in range(20):
    with torch.no_grad(): _=model(x1,c1)
ts=[]
for _ in range(200):
    a=time.perf_counter()
    with torch.no_grad(): _=model(x1,c1)
    ts.append((time.perf_counter()-a)*1000)

print(f"\n{'='*55}\nHIERARCHICAL — EVAL SCORECARD\n{'='*55}")
print(f"  Overall:  {ov:.3f} | QUIC: {qa:.3f} | TLS: {ta:.3f}")
print(f"  Fine intra: {fine_intra:.3f} (target >0.7) | Fine inter: {fine_inter:.3f} (target <0.3)")
print(f"  Super-group acc: {sg_acc:.3f}  | SG inter-cosine: {sg_inter:.3f}")
print(f"  Latency: {np.median(ts):.2f}ms")
print(f"\n  Per-class TLS accuracy:")
for c in range(NC):
    m=t&(te_y==c)
    if m.sum(): print(f"    {CATS[c]:18s} (SG={SUPERGROUP_MAP[CATS[c]]}) {(pred_c[m]==c).mean():.3f}")
print(f"\n  Super-group per-group accuracy:")
for g in range(N_SG):
    m=te_sg==g; members=[CATS[i] for i in range(NC) if SG[i]==g]
    print(f"    G{g} {str(members)[:40]:40s}: {(pred_sg[m]==g).mean():.3f}")

try:
    from sklearn.manifold import TSNE; import matplotlib.pyplot as plt
    ids=np.concatenate([rng.choice(np.where(te_y==c)[0],min(200,(te_y==c).sum()),replace=False) for c in range(NC)])
    emb=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(zte[ids])
    yy=te_y[ids]; pp2=te_p[ids]; cmap=plt.cm.tab10

    # fine-class t-SNE
    plt.figure(figsize=(10,8))
    for c in range(NC):
        mq=(yy==c)&(pp2=="quic"); mt=(yy==c)&(pp2=="tls")
        if mq.sum(): plt.scatter(emb[mq,0],emb[mq,1],s=12,marker="o",alpha=0.5,color=cmap(c%10),label=CATS[c])
        if mt.sum(): plt.scatter(emb[mt,0],emb[mt,1],s=24,marker="^",alpha=0.8,edgecolors="k",linewidths=0.3,color=cmap(c%10))
    plt.title(f"HIERARCHICAL — Fine class | QUIC ○ / TLS △ | TLS={ta:.3f} inter={fine_inter:.3f}")
    plt.legend(fontsize=7,ncol=2); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(str(Path(OUT,"hierarchical_tsne_fine.png")),dpi=130,bbox_inches="tight")

    # super-group t-SNE
    sg_colors=plt.cm.Set1
    sg_names={0:"ecommerce+gaming",1:"messaging+social",2:"file+cdn+video+search",3:"audio"}
    plt.figure(figsize=(10,8))
    for g in range(N_SG):
        mg=SG[yy]==g; plt.scatter(emb[mg,0],emb[mg,1],s=14,alpha=0.6,color=sg_colors(g),label=f"G{g}: {sg_names[g]}")
    plt.title(f"HIERARCHICAL — Super-group | sg_acc={sg_acc:.3f} sg_inter={sg_inter:.3f}")
    plt.legend(fontsize=8); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(str(Path(OUT,"hierarchical_tsne_sg.png")),dpi=130,bbox_inches="tight")
    print(f"[saved] t-SNE plots in {OUT}/")
except Exception as e: print("[tsne skipped]",e)

result={"model":"hierarchical","overall":float(ov),"quic":float(qa),"tls":float(ta),
        "fine_intra":float(fine_intra),"fine_inter":float(fine_inter),
        "sg_acc":float(sg_acc),"sg_inter":float(sg_inter),"latency_ms":float(np.median(ts)),
        "per_class_tls":{CATS[c]:float((pred_c[t&(te_y==c)]==c).mean()) for c in range(NC) if (t&(te_y==c)).sum()},
        "per_sg":{f"G{g}":{CATS[i]:None for i in range(NC) if SG[i]==g} for g in range(N_SG)}}
Path(f"{OUT}/hierarchical_eval.json").write_text(json.dumps(result,indent=2))
print(f"[saved] {OUT}/hierarchical_eval.json")
