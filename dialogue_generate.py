import torch, torch.nn as nn, torch.nn.functional as F, math
dev="mps"; d=64; K=6
ck=torch.load("dialogue_model.pt")
vocab=ck['vocab']; V=len(vocab); w2i={w:i for i,w in enumerate(vocab)}
class Mod(nn.Module):
    def __init__(s):
        super().__init__(); s.E=nn.Embedding(V,d)
        s.Wq=nn.Linear(d,d,bias=False);s.Wk=nn.Linear(d,d,bias=False);s.Wv=nn.Linear(d,d,bias=False)
        s.head=nn.Linear(2*d,V,bias=False)
    def forward(s,ctx):
        emb=s.E(ctx);cur=emb[:,0];q=s.Wq(cur).unsqueeze(1);k=s.Wk(emb);v=s.Wv(emb)
        a=torch.softmax((q*k).sum(-1)/math.sqrt(d),1);mix=(a.unsqueeze(-1)*v).sum(1)
        return s.head(torch.cat([cur,F.relu(mix)],-1))
m=Mod().to(dev); m.load_state_dict(ck['sd']); m.eval()
@torch.no_grad()
def reply(prompt,n=30,temp=0.4):
    toks=["<user>"]+prompt.lower().split()+["<asst>"]; idl=[w2i.get(w,1) for w in toks]
    while len(idl)<K: idl=[w2i["<user>"]]+idl
    out=[]
    for _ in range(n):
        ctx=torch.tensor([[idl[-1-j] for j in range(K)]],device=dev)
        p=torch.softmax(m(ctx)[0]/temp,0).cpu()          # CPU multinomial (MPS bug fix)
        nx=int(torch.multinomial(p,1).item())
        if vocab[nx] in ("<user>","<pad>","<asst>"): break
        idl.append(nx); out.append(vocab[nx])
    return " ".join(out) if out else "(empty)"
for q in ["i feel so happy today","i lost my job yesterday","my dog is sick","i am scared about my exam","thank you so much"]:
    print(f"USER: {q}"); print(f"BOT : {reply(q)}\n")
