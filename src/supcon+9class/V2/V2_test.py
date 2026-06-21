# ============================================================================
# V2 FiLM — PUSH MODEL + FULL EVAL (cosine + t-SNE) + PUSH RESULTS
# ============================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download, HfApi, login
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
api=HfApi()
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HF="donbosoc/shigan-mjkan-baseline"
V2DIR=os.path.join(os.getenv("OUT_DIR","./checkpoints"),"v2"); Path(V2DIR).mkdir(parents=True,exist_ok=True)

# ============ STEP 1: load V2 checkpoint ============
if os.getenv("MJKAN_FROM_HF") == "1":
    ckl=torch.load(hf_hub_download(HF,"protocol_invariance/v2_film/supcon_v2_best.pt"),map_location="cpu",weights_only=False)
else:
    ckl=torch.load(f"{V2DIR}/v2_film_best.pt",map_location="cpu",weights_only=False)
v2res={"experiment":"V2 — FiLM conditioning on 6-feature flow-context (the deck's MJKAN component)",
       "best_epoch":int(ckl.get("epoch",-1)),
       "tls_test_acc":float(ckl.get("tls_acc",-1)),"quic_test_acc":float(ckl.get("quic_acc",-1)),
       "overall_test_acc":float(ckl.get("overall",-1)),
       "comparison":{"baseline":0.302,"V1":0.720,"V3":0.715,"V5z":0.734,"V2_FiLM":float(ckl.get("tls_acc",-1))},
       "key_finding":"FiLM conditioning is the BEST variant (TLS 0.755). Notably lifts messaging (0.55->~0.70) by "
                     "conditioning on flow-context that distinguishes videoconferencing from text messaging. "
                     "cdn_web_assets remains stuck (genuine bulk-download overlap, no context signal separates it)."}
Path(f"{V2DIR}/v2_result.json").write_text(json.dumps(v2res,indent=2))
if os.getenv("MJKAN_PUSH_HF") == "1":
    for local,remote in [(f"{V2DIR}/v2_film_best.pt","protocol_invariance/v2_film/supcon_v2_best.pt"),
                         (f"{V2DIR}/v2_result.json","protocol_invariance/v2_film/v2_result.json")]:
        api.upload_file(path_or_fileobj=local,path_in_repo=remote,repo_id=HF)
        print("[pushed]",remote)

# ============ STEP 2: full eval (must rebuild FiLM model arch) ============
meta=json.load(open(hf_hub_download(HF,"protocol_invariance/data/joint_meta.json")))
CATS=meta["categories"]; NC=meta["num_classes"]; N_CTX=meta["n_ctx_feats"]

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
    def __init__(s,n_ctx,dm,hidden=64):
        super().__init__();s.net=nn.Sequential(nn.Linear(n_ctx,hidden),nn.ReLU(),nn.Linear(hidden,2*dm));s.dm=dm
        nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)
    def forward(s,h,ctx):
        gb=s.net(ctx); g,b=gb[:,:s.dm],gb[:,s.dm:]; return (1.0+g)*h+b
class FiLMSupConModel(nn.Module):
    def __init__(s,nf=5,n_ctx=6,dm=128,nl=2,ds=16,dp=0.1):
        super().__init__();s.encoder=Encoder(nf,dm,nl,ds,dp);s.film=FiLM(n_ctx,dm)
        s.proj=nn.Sequential(nn.Linear(dm,dm),nn.BatchNorm1d(dm),nn.ReLU(),nn.Linear(dm,dm))
    def forward(s,seq,ctx): h=s.film(s.encoder(seq),ctx); return h,F.normalize(s.proj(h),dim=1)

model=FiLMSupConModel(meta["n_seq_feats"],N_CTX,128,2,16,0.1).to(DEVICE)
model.load_state_dict(ckl["model"]); model.eval()

dt=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_test.npz"),allow_pickle=True)
dtr=np.load(hf_hub_download(HF,"protocol_invariance/data/joint_train.npz"),allow_pickle=True)
te_seq=dt["seq"]; te_ctx=dt["ctx"]; te_y=dt["label"].astype(int); te_p=dt["protocol"].astype(str)
tr_seq=dtr["seq"]; tr_ctx=dtr["ctx"]; tr_y=dtr["label"].astype(int)

@torch.no_grad()
def embed(seq,ctx,bs=512):
    Z=[]
    for i in range(0,len(seq),bs):
        sb=torch.from_numpy(seq[i:i+bs]).float().to(DEVICE); cb=torch.from_numpy(ctx[i:i+bs]).float().to(DEVICE)
        Z.append(model(sb,cb)[1].cpu().numpy())
    return np.concatenate(Z)
def knn(rz,ry,qz,k=20):
    out=[]
    for i in range(0,len(qz),2048):
        s=qz[i:i+2048]@rz.T; idx=np.argpartition(-s,k,1)[:,:k]
        out.append(np.array([np.bincount(ry[v],minlength=NC).argmax() for v in idx]))
    return np.concatenate(out)

ztr=embed(tr_seq,tr_ctx); zte=embed(te_seq,te_ctx)
rng=np.random.default_rng(0)
ri=np.concatenate([rng.choice(np.where(tr_y==c)[0],min(2000,(tr_y==c).sum()),replace=False) for c in range(NC)])
pred=knn(ztr[ri],tr_y[ri],zte)
q=te_p=="quic"; t=te_p=="tls"
ov=(pred==te_y).mean(); qa=(pred[q]==te_y[q]).mean(); ta=(pred[t]==te_y[t]).mean()
cents=np.stack([zte[te_y==c].mean(0) for c in range(NC)]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
intra=np.mean([(zte[te_y==c]@cents[c]).mean() for c in range(NC)])
C=cents@cents.T; inter=(C.sum()-NC)/(NC*NC-NC)
perclass={CATS[c]:float((pred[t&(te_y==c)]==c).mean()) for c in range(NC) if (t&(te_y==c)).sum()}
print(f"[V2 FiLM] overall={ov:.3f} QUIC={qa:.3f} TLS={ta:.3f} | intra={intra:.3f} inter={inter:.3f}")
for c,v in perclass.items(): print(f"   {c:18s} {v:.3f}")

full={"overall":float(ov),"quic":float(qa),"tls":float(ta),"intra":float(intra),"inter":float(inter),
      "per_class_tls":perclass}
Path(f"{V2DIR}/v2_full_eval.json").write_text(json.dumps(full,indent=2))

# ============ STEP 3: t-SNE ============
from sklearn.manifold import TSNE; import matplotlib.pyplot as plt
ids=np.concatenate([rng.choice(np.where(te_y==c)[0],min(250,(te_y==c).sum()),replace=False) for c in range(NC)])
emb=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(zte[ids]); yy=te_y[ids]; pp=te_p[ids]
plt.figure(figsize=(10,8)); cmap=plt.cm.tab10
for c in range(NC):
    mq=(yy==c)&(pp=="quic"); mt=(yy==c)&(pp=="tls")
    if mq.sum(): plt.scatter(emb[mq,0],emb[mq,1],s=12,marker="o",alpha=0.5,color=cmap(c%10),label=CATS[c])
    if mt.sum(): plt.scatter(emb[mt,0],emb[mt,1],s=24,marker="^",alpha=0.8,edgecolors="k",linewidths=0.3,color=cmap(c%10))
plt.title(f"V2 FiLM (best model) — QUIC ○ / TLS △  |  TLS={ta:.3f} inter={inter:.3f}")
plt.legend(fontsize=7,ncol=2,markerscale=1.2); plt.xticks([]); plt.yticks([])
plt.tight_layout(); plt.savefig(f"{V2DIR}/v2_tsne.png",dpi=130,bbox_inches="tight"); plt.show()

if os.getenv("MJKAN_PUSH_HF") == "1":
    for local,remote in [(f"{V2DIR}/v2_full_eval.json","protocol_invariance/v2_film/v2_full_eval.json"),
                         (f"{V2DIR}/v2_tsne.png","protocol_invariance/v2_film/v2_tsne.png")]:
        api.upload_file(path_or_fileobj=local,path_in_repo=remote,repo_id=HF)
        print("[pushed]",remote)
    print(f"\nV2 FiLM fully archived -> https://huggingface.co/{HF}/tree/main/protocol_invariance/v2_film")