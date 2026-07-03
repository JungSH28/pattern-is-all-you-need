# NEW GOAL: simple conversation. dialogue data (empathetic) + turn markers + reply generation.
import torch, torch.nn as nn, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset
dev="mps"; d=64; K=6; EP=6; MAXV=8000
ds=load_dataset('Estwld/empathetic_dialogues_llm', split='train')
SP=["<pad>","<unk>","<user>","<asst>"]
# build token stream with turn markers
def conv_tokens(conv):
    toks=[]
    for turn in conv:
        toks.append("<user>" if turn['role']=='user' else "<asst>")
        toks+=turn['content'].lower().split()
    return toks
cnt=Counter()
streams=[]
for ex in ds:
    t=conv_tokens(ex['conversations']); streams.append(t); cnt.update(t)
vocab=SP+[w for w,_ in cnt.most_common(MAXV-len(SP)) if w not in SP]
w2i={w:i for i,w in enumerate(vocab)}
ids=[]
for t in streams: ids+=[w2i.get(w,1) for w in t]
print(f"convs={len(streams)} tokens={len(ids)} vocab={len(vocab)}",flush=True)
V=len(vocab)
# chunk into training windows
SEQ=24
data=torch.tensor(ids,dtype=torch.long)
nb=(len(data)-1)//SEQ
data=data[:nb*SEQ].view(nb,SEQ)
class Mod(nn.Module):
    def __init__(s):
        super().__init__(); s.E=nn.Embedding(V,d)
        s.Wq=nn.Linear(d,d,bias=False);s.Wk=nn.Linear(d,d,bias=False);s.Wv=nn.Linear(d,d,bias=False)
        for p in [*s.Wq.parameters(),*s.Wk.parameters(),*s.Wv.parameters()]: p.requires_grad=False
        s.head=nn.Linear(2*d,V,bias=False)
    def forward(s,ctx):
        emb=s.E(ctx);cur=emb[:,0]
        q=s.Wq(cur).unsqueeze(1);k=s.Wk(emb);v=s.Wv(emb)
        a=torch.softmax((q*k).sum(-1)/math.sqrt(d),1);mix=(a.unsqueeze(-1)*v).sum(1)
        return s.head(torch.cat([cur,F.relu(mix)],-1))
m=Mod().to(dev); opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3)
BS=64
for ep in range(EP):
    m.train(); perm=torch.randperm(nb)
    for i in range(0,nb-BS,BS):
        b=data[perm[i:i+BS]].to(dev)  # (BS,SEQ)
        for p in range(K,SEQ):
            ctx=torch.stack([b[:,p-1-j] for j in range(K)],1); y=b[:,p]
            opt.zero_grad(); F.cross_entropy(m(ctx),y).backward(); opt.step()
    print(f"ep{ep+1} done",flush=True)
m.eval()
@torch.no_grad()
def reply(prompt, n=30, temp=0.7):
    toks=["<user>"]+prompt.lower().split()+["<asst>"]
    ids=[w2i.get(w,1) for w in toks]
    while len(ids)<K: ids=[w2i["<user>"]]+ids
    out=[]
    for _ in range(n):
        ctx=torch.tensor([[ids[-1-j] for j in range(K)]],device=dev)
        p=torch.softmax(m(ctx)[0]/temp,0); nx=torch.multinomial(p,1).item()
        if vocab[nx]=="<user>": break
        ids.append(nx); out.append(vocab[nx])
    return " ".join(out)
print("--- conversation ---",flush=True)
for q in ["i feel so happy today","i lost my job yesterday","my dog is sick","what should i do"]:
    print(f"USER: {q}\nBOT : {reply(q)}\n",flush=True)
