# Model reconstruction per design principles: sparse as GENETIC PRIOR + VARIABLE k (adaptive assembly).
#   threshold firing (fire if activation > theta; theta homeostatic to target rate rho)
#   -> k = #active VARIES per input = adaptive assembly size (harder/stronger input recruits more).
#   fixed random routing (reservoir prior), learned E/head. bio: threshold neuron + homeostasis.
# Tests: mean density, density VARIANCE (does k vary?), corr(k, difficulty) [intelligence hypothesis],
#        val ppl, conversation. vs dense and fixed top-k.
import torch, torch.nn as nn, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
d,K,EP,MAXV,SEQ=64,6,10,6000,24
ds=load_dataset('Estwld/empathetic_dialogues_llm',split='train')
SP=["<pad>","<unk>","<user>","<asst>"]
def toks_of(c):
    t=[]
    for tn in c: t.append("<user>" if tn['role']=='user' else "<asst>"); t+=tn['content'].lower().split()
    return t
streams=[toks_of(ex['conversations']) for ex in ds]; nval=len(streams)//20
cnt=Counter()
for t in streams[nval:]: cnt.update(t)
vocab=SP+[w for w,_ in cnt.most_common(MAXV-4) if w not in SP]; w2i={w:i for i,w in enumerate(vocab)}; V=len(vocab)
def enc(sl):
    ids=[]
    for t in sl: ids+=[w2i.get(w,1) for w in t]
    x=torch.tensor(ids);nb=(len(x)-1)//SEQ;return x[:nb*SEQ].view(nb,SEQ)
tr,va=enc(streams[nval:]),enc(streams[:nval])
print(f"V={V}",flush=True)
class Mod(nn.Module):
    def __init__(s,mode,rho=0.10):     # rho=target firing rate (genetic setpoint)
        super().__init__(); s.mode=mode; s.rho=rho; s.theta=0.3
        s.E=nn.Embedding(V,d)
        s.Wq=nn.Linear(d,d,bias=False);s.Wk=nn.Linear(d,d,bias=False);s.Wv=nn.Linear(d,d,bias=False)
        for p in [*s.Wq.parameters(),*s.Wk.parameters(),*s.Wv.parameters()]: p.requires_grad=False
        s.head=nn.Linear(2*d,V,bias=False)
    def act(s,feat,train=False):
        if s.mode=="dense": return feat
        if s.mode=="topk":
            v,i=feat.topk(8,1);o=torch.zeros_like(feat);o.scatter_(1,i,v);return o
        # adaptive threshold (variable k): fire if > theta; homeostasis on theta
        mask=(feat>s.theta).float()
        if train:
            rate=mask.mean().item(); s.theta+=0.01*(rate-s.rho)
        return feat*mask
    def forward(s,ctx,train=False):
        e=s.E(ctx);cur=e[:,0];q=s.Wq(cur).unsqueeze(1);k=s.Wk(e);v=s.Wv(e)
        a=torch.softmax((q*k).sum(-1)/math.sqrt(d),1);mix=(a.unsqueeze(-1)*v).sum(1)
        feat=torch.cat([F.relu(cur),F.relu(mix)],1)
        feat=s.act(feat,train)
        return s.head(feat), feat
def run(mode,tag):
    torch.manual_seed(0); m=Mod(mode).to(dev)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3);BS=64
    for ep in range(EP):
        perm=torch.randperm(len(tr))
        for i in range(0,len(tr)-BS,BS):
            b=tr[perm[i:i+BS]].to(dev)
            for p in range(K,SEQ):
                ctx=torch.stack([b[:,p-1-j] for j in range(K)],1)
                opt.zero_grad();lg,_=m(ctx,train=True);F.cross_entropy(lg,b[:,p]).backward();opt.step()
    m.eval();tot=0;c=0;ks=[];diffs=[]
    with torch.no_grad():
        for i in range(0,len(va)-64,64):
            b=va[i:i+64].to(dev)
            for p in range(K,SEQ):
                ctx=torch.stack([b[:,p-1-j] for j in range(K)],1);lg,feat=m(ctx)
                l=F.cross_entropy(lg,b[:,p],reduction='none')
                tot+=l.sum().item();c+=b.size(0)
                ks+=(feat.abs()>1e-4).float().mean(1).tolist()   # per-sample density (k/2d)
                diffs+=l.tolist()                                 # per-sample difficulty (loss)
    ppl=math.exp(tot/c); ks=torch.tensor(ks);diffs=torch.tensor(diffs)
    corr=torch.corrcoef(torch.stack([ks,diffs]))[0,1].item() if mode=="adapt" else float('nan')
    @torch.no_grad()
    def reply(q,temp=0.4,n=20):
        idl=[w2i.get(w,1) for w in ["<user>"]+q.lower().split()+["<asst>"]]
        while len(idl)<K: idl=[w2i["<user>"]]+idl
        out=[]
        for _ in range(n):
            ctx=torch.tensor([[idl[-1-j] for j in range(K)]],device=dev);lg,_=m(ctx)
            pp=torch.softmax(lg[0]/temp,0).cpu();nx=int(torch.multinomial(pp,1).item())
            if vocab[nx] in ("<user>","<pad>","<asst>"): break
            idl.append(nx);out.append(vocab[nx])
        return " ".join(out) if out else "(empty)"
    print(f"\n== {tag} ppl={ppl:.1f} density mean={ks.mean():.3f} std={ks.std():.3f} corr(k,difficulty)={corr:+.3f} theta={getattr(m,'theta',0):.2f} ==",flush=True)
    torch.manual_seed(1)
    for q in ["i am really sad","my dog is sick","i got promoted at work"]:
        print(f"USER: {q}\nBOT : {reply(q)}",flush=True)
print("(dense ppl~105 dens0.37 | fixed top-k=8: dens0.06)",flush=True)
run("dense","DENSE")
run("topk","FIXED-TOPK")
run("adapt","ADAPTIVE-VARIABLE-K")
