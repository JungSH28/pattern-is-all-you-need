# GOAL loop 2: real attention (dot-score + softmax routing) vs additive. same dims. target->53.
import torch, torch.nn as nn, torch.nn.functional as F, math
from data import load_wikitext2
dev="mps"; VOCAB=2000; d=64; K=8; EP=12
tb,vb,vocab,w2i=load_wikitext2(max_vocab=VOCAB,seq_len=16,batch_size=32)
class Attn(nn.Module):
    def __init__(s,mode):
        super().__init__(); s.mode=mode
        s.E=nn.Embedding(VOCAB,d)
        s.Wq=nn.Linear(d,d,bias=False); s.Wk=nn.Linear(d,d,bias=False); s.Wv=nn.Linear(d,d,bias=False)
        s.U=nn.ParameterList([nn.Parameter(torch.randn(d,d)*0.1) for _ in range(K)]) # additive path
        s.head=nn.Linear(2*d,VOCAB,bias=False)
    def forward(s,ctx):  # ctx (B,K): ctx[:,0]=t ... ctx[:,K-1]=t-K+1
        emb=s.E(ctx)                       # (B,K,d)
        cur=emb[:,0]
        if s.mode=="attn":
            q=s.Wq(cur).unsqueeze(1)       # (B,1,d)
            k=s.Wk(emb); v=s.Wv(emb)       # (B,K,d)
            a=torch.softmax((q*k).sum(-1)/math.sqrt(d),dim=1)  # (B,K)
            mix=(a.unsqueeze(-1)*v).sum(1) # (B,d)
        else:
            mix=sum(emb[:,j]@s.U[j] for j in range(K))  # additive
        return s.head(torch.cat([cur,F.relu(mix)],-1))
def batches(split):
    out=[]
    for inp,_ in split:
        T=inp.shape[1]
        for p in range(K,T):
            out.append((torch.stack([inp[:,p-1-j] for j in range(K)],1),inp[:,p]))
    return out
trb=batches(tb); vbb=batches(vb)
def run(mode):
    torch.manual_seed(0); m=Attn(mode).to(dev)
    npar=sum(p.numel() for p in m.parameters()); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    best=1e9
    for ep in range(1,EP+1):
        m.train()
        for ctx,y in trb:
            ctx=ctx.to(dev);y=y.to(dev); opt.zero_grad()
            F.cross_entropy(m(ctx),y).backward(); opt.step()
        m.eval(); tot=0;cnt=0
        with torch.no_grad():
            for ctx,y in vbb:
                ctx=ctx.to(dev);y=y.to(dev); tot+=F.cross_entropy(m(ctx),y,reduction='sum').item();cnt+=len(y)
        best=min(best,math.exp(tot/cnt))
    print(f"{mode:8s} params={npar/1e6:.2f}M BEST={best:.1f}",flush=True)
print("tf 0.42M 53 | additive control | GOAL attn beats additive & ->53",flush=True)
run("add"); run("attn")
