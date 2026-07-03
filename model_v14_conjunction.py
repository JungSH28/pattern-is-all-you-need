# v14: learned embedding + nonlinear conjunction + bio credit (FA). Unifies session.
#   h = relu(E[cur]@U1 + E[prev]@U2)  (order-preserving conjunction, LEARNED)
#   logits = h@W
# credit: backprop (ceiling) vs feedback-alignment (bio). vs fixed count-63.
import torch, torch.nn.functional as F
from data import load_wikitext2
dev="mps"; VOCAB=2000; d=96; M=512; EP=12; lr=0.03
tb,vb,vocab,w2i=load_wikitext2(max_vocab=VOCAB,seq_len=16,batch_size=32)
def run(mode):
    torch.manual_seed(0)
    E=torch.randn(VOCAB,d,device=dev)*0.1
    U1=torch.randn(d,M,device=dev)*0.1; U2=torch.randn(d,M,device=dev)*0.1
    W=torch.randn(M,VOCAB,device=dev)*0.1
    B=torch.randn(VOCAB,M,device=dev)*0.1   # FA feedback for top layer
    best=1e9
    for ep in range(1,EP+1):
        for inp,_ in tb:
            inp=inp.to(dev)
            cur=inp[:,1:-1].reshape(-1); prev=inp[:,:-2].reshape(-1); nxt=inp[:,2:].reshape(-1)
            msk=nxt!=0; cur=cur[msk]; prev=prev[msk]; nxt=nxt[msk]; P=len(nxt)
            ec=E[cur]; ep_=E[prev]
            z=ec@U1+ep_@U2; h=F.relu(z)
            logits=h@W; p=torch.softmax(logits,1)
            e=p; e[torch.arange(P,device=dev),nxt]-=1
            gW=h.t()@e/P
            fb = W.t() if mode=="backprop" else B
            dh=(e@fb)*(z>0).float()          # credit to hidden
            gU1=ec.t()@dh/P; gU2=ep_.t()@dh/P
            dEc=dh@U1.t(); dEp=dh@U2.t()      # credit to embedding (via U.t; local-ish)
            W-=lr*gW; U1-=lr*gU1; U2-=lr*gU2
            E.index_add_(0,cur,-lr*dEc/P); E.index_add_(0,prev,-lr*dEp/P)
        tot,n=0,0
        for inp,_ in vb[:200]:
            inp=inp.to(dev);c=inp[:,1:-1].reshape(-1);pv=inp[:,:-2].reshape(-1);nx=inp[:,2:].reshape(-1);mm=nx!=0
            hh=F.relu(E[c[mm]]@U1+E[pv[mm]]@U2)
            tot+=F.cross_entropy(hh@W,nx[mm]).item();n+=1
        ppl=2.718**min(tot/n,700.0); best=min(best,ppl)
        if ep in (1,4,8,12): print(f"{mode:8s} ep{ep} ppl={ppl:.1f}",flush=True)
    print(f"  -> {mode} BEST={best:.1f}",flush=True)
print("fixed baselines: count-bigram 63, our-best 63, backprop-nn(no conj) 57",flush=True)

run("feedback")
