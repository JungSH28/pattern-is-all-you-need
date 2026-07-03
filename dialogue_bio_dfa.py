# FULLY BIO dialogue: DFA credit (fixed random feedback, no backprop/autograd). E+head learned local.
import torch, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset
dev="mps"; d=64; K=6; EP=8; MAXV=6000; SEQ=24; lr=0.05
ds=load_dataset('Estwld/empathetic_dialogues_llm', split='train')
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
tr=enc(streams[nval:]); va=enc(streams[:nval])
print(f"V={V} train_win={len(tr)}",flush=True)
g=torch.Generator().manual_seed(0)
E=(torch.randn(V,d,generator=g)*0.1).to(dev)
Wq=(torch.randn(d,d,generator=g)*0.1).to(dev);Wk=(torch.randn(d,d,generator=g)*0.1).to(dev);Wv=(torch.randn(d,d,generator=g)*0.1).to(dev)
head=(torch.randn(2*d,V,generator=g)*0.1).to(dev)
B_E=(torch.randn(V,d,generator=g)*0.1).to(dev)   # DFA fixed feedback to embedding
def fwd(ctx):
    emb=E[ctx];cur=emb[:,0]
    q=cur@Wq;k=emb@Wk;v=emb@Wv
    a=torch.softmax((q.unsqueeze(1)*k).sum(-1)/math.sqrt(d),1)
    mix=(a.unsqueeze(-1)*v).sum(1)
    feat=torch.cat([cur,F.relu(mix)],1)
    return feat, feat@head
@torch.no_grad()
def valppl():
    tot=0;c=0
    for i in range(0,len(va)-64,64):
        b=va[i:i+64].to(dev)
        for p in range(K,SEQ):
            ctx=torch.stack([b[:,p-1-j] for j in range(K)],1); _,lg=fwd(ctx)
            tot+=F.cross_entropy(lg,b[:,p],reduction='sum').item();c+=b.size(0)
    return math.exp(tot/c)
BS=64
for ep in range(EP):
    perm=torch.randperm(len(tr))
    for i in range(0,len(tr)-BS,BS):
        b=tr[perm[i:i+BS]].to(dev)
        for p in range(K,SEQ):
            ctx=torch.stack([b[:,p-1-j] for j in range(K)],1); y=b[:,p]
            feat,lg=fwd(ctx); P=len(y)
            pr=torch.softmax(lg,1); e=pr.clone(); e[torch.arange(P,device=dev),y]-=1
            head-=lr*(feat.t()@e)/P                    # local delta (exact for head)
            dsig=(e@B_E)/P                              # DFA random feedback -> embedding
            for j in range(K): E.index_add_(0,ctx[:,j],-lr*dsig)
    print(f"ep{ep+1} val_ppl={valppl():.1f}",flush=True)
# generate
@torch.no_grad()
def reply(prompt,n=30,temp=0.7):
    toks=["<user>"]+prompt.lower().split()+["<asst>"]; idl=[w2i.get(w,1) for w in toks]
    while len(idl)<K: idl=[w2i["<user>"]]+idl
    out=[]
    for _ in range(n):
        ctx=torch.tensor([[idl[-1-j] for j in range(K)]],device=dev); _,lg=fwd(ctx)
        pp=torch.softmax(lg[0]/temp,0).cpu(); nx=int(torch.multinomial(pp,1).item())
        if vocab[nx] in ("<user>","<pad>","<asst>"): break
        idl.append(nx); out.append(vocab[nx])
    return " ".join(out) if out else "(empty)"
print("--- BIO(DFA) conversation ---",flush=True)
for q in ["i feel so happy today","my dog is sick","i lost my job yesterday","i am scared about my exam"]:
    print(f"USER: {q}\nBOT : {reply(q)}",flush=True)
print("(backprop baseline was val ppl 96.8)",flush=True)
