# Recover the sparse principle: enforce top-k active hidden (sparse coding; ~brain firing rate).
# Does the dialogue model stay conversational when SPARSE? measures density + val ppl + samples.
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
def topk_sparse(x,k):     # keep top-k per row, zero rest (sparse coding / k-WTA on feature)
    if k>=x.shape[1]: return x
    val,idx=x.topk(k,dim=1); out=torch.zeros_like(x); out.scatter_(1,idx,val); return out
class Mod(nn.Module):
    def __init__(s,ksparse):
        super().__init__(); s.k=ksparse
        s.E=nn.Embedding(V,d)
        s.Wq=nn.Linear(d,d,bias=False);s.Wk=nn.Linear(d,d,bias=False);s.Wv=nn.Linear(d,d,bias=False)
        for p in [*s.Wq.parameters(),*s.Wk.parameters(),*s.Wv.parameters()]: p.requires_grad=False
        s.head=nn.Linear(2*d,V,bias=False)
    def forward(s,ctx):
        e=s.E(ctx);cur=e[:,0];q=s.Wq(cur).unsqueeze(1);k=s.Wk(e);v=s.Wv(e)
        a=torch.softmax((q*k).sum(-1)/math.sqrt(d),1);mix=(a.unsqueeze(-1)*v).sum(1)
        feat=torch.cat([cur,F.relu(mix)],1)
        feat=topk_sparse(feat,s.k)          # SPARSE: only top-k of 2d active
        return s.head(feat), feat
def batches(sp):
    o=[]
    for x in [sp]:
        for r in range(len(x)):
            b=x[r]
    return None
def run(ksparse,tag):
    torch.manual_seed(0); m=Mod(ksparse).to(dev)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3);BS=64
    for ep in range(EP):
        perm=torch.randperm(len(tr))
        for i in range(0,len(tr)-BS,BS):
            b=tr[perm[i:i+BS]].to(dev)
            for p in range(K,SEQ):
                ctx=torch.stack([b[:,p-1-j] for j in range(K)],1)
                opt.zero_grad();lg,_=m(ctx);F.cross_entropy(lg,b[:,p]).backward();opt.step()
    m.eval();tot=0;c=0;dens=0;nb=0
    with torch.no_grad():
        for i in range(0,len(va)-64,64):
            b=va[i:i+64].to(dev)
            for p in range(K,SEQ):
                ctx=torch.stack([b[:,p-1-j] for j in range(K)],1);lg,feat=m(ctx)
                tot+=F.cross_entropy(lg,b[:,p],reduction='sum').item();c+=b.size(0)
                dens+=(feat.abs()>1e-4).float().mean().item();nb+=1
    ppl=math.exp(tot/c)
    @torch.no_grad()
    def reply(prompt,temp=0.4,n=20):
        idl=[w2i.get(w,1) for w in ["<user>"]+prompt.lower().split()+["<asst>"]]
        while len(idl)<K: idl=[w2i["<user>"]]+idl
        out=[]
        for _ in range(n):
            ctx=torch.tensor([[idl[-1-j] for j in range(K)]],device=dev);lg,_=m(ctx)
            pp=torch.softmax(lg[0]/temp,0).cpu();nx=int(torch.multinomial(pp,1).item())
            if vocab[nx] in ("<user>","<pad>","<asst>"): break
            idl.append(nx);out.append(vocab[nx])
        return " ".join(out) if out else "(empty)"
    print(f"\n== {tag} (k={ksparse}/{2*d}) val_ppl={ppl:.1f} density={dens/nb:.2f} ==",flush=True)
    torch.manual_seed(1)
    for q in ["i am really sad","my dog is sick","i got promoted at work"]:
        print(f"USER: {q}\nBOT : {reply(q)}",flush=True)
print("(dense baseline: ppl 96.8, density 0.74)",flush=True)
run(2*d,"DENSE")     # no sparsity (control)
run(16,"SPARSE-16")  # 16 of 128 active (~12%)
run(8,"SPARSE-8")    # 8 of 128 (~6%, brain-like)
