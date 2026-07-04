# Close the fully-bio gap: STRUCTURED (sign-concordant) DFA feedback vs random.
#   random  : B_E fixed random (baseline, val ppl ~160).
#   sign    : feedback = sign of the direct readout path head[:d] (no magnitude transport;
#             E/I-consistent; session found sign-concordant > random).
# No backprop/autograd. Learned = E, head. Routing fixed random. Compares val ppl + samples.
import torch, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset

dev = "mps" if torch.backends.mps.is_available() else "cpu"
d, K, EP, MAXV, SEQ, lr = 64, 6, 12, 6000, 24, 0.05
ds = load_dataset('Estwld/empathetic_dialogues_llm', split='train')
SP = ["<pad>", "<unk>", "<user>", "<asst>"]


def toks_of(c):
    t = []
    for tn in c:
        t.append("<user>" if tn['role'] == 'user' else "<asst>")
        t += tn['content'].lower().split()
    return t


streams = [toks_of(ex['conversations']) for ex in ds]
nval = len(streams) // 20
cnt = Counter()
for t in streams[nval:]:
    cnt.update(t)
vocab = SP + [w for w, _ in cnt.most_common(MAXV - 4) if w not in SP]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)


def enc(sl):
    ids = []
    for t in sl:
        ids += [w2i.get(w, 1) for w in t]
    x = torch.tensor(ids)
    nb = (len(x) - 1) // SEQ
    return x[:nb * SEQ].view(nb, SEQ)


tr, va = enc(streams[nval:]), enc(streams[:nval])
print(f"V={V} train_win={len(tr)}", flush=True)


def run(mode):
    g = torch.Generator().manual_seed(0)
    E = (torch.randn(V, d, generator=g) * 0.1).to(dev)
    Wq = (torch.randn(d, d, generator=g) * 0.1).to(dev)
    Wk = (torch.randn(d, d, generator=g) * 0.1).to(dev)
    Wv = (torch.randn(d, d, generator=g) * 0.1).to(dev)
    head = (torch.randn(2 * d, V, generator=g) * 0.1).to(dev)
    B_rand = (torch.randn(V, d, generator=g) * 0.1).to(dev)

    def fwd(ctx):
        emb = E[ctx]; cur = emb[:, 0]
        q, k, v = cur @ Wq, emb @ Wk, emb @ Wv
        a = torch.softmax((q.unsqueeze(1) * k).sum(-1) / math.sqrt(d), 1)
        mix = (a.unsqueeze(-1) * v).sum(1)
        return torch.cat([cur, F.relu(mix)], 1), None

    def logits(feat):
        return feat @ head

    @torch.no_grad()
    def valppl():
        tot, c = 0.0, 0
        for i in range(0, len(va) - 64, 64):
            b = va[i:i + 64].to(dev)
            for p in range(K, SEQ):
                ctx = torch.stack([b[:, p - 1 - j] for j in range(K)], 1)
                feat, _ = fwd(ctx)
                tot += F.cross_entropy(logits(feat), b[:, p], reduction='sum').item(); c += b.size(0)
        return math.exp(tot / c)

    BS = 64
    for ep in range(EP):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr) - BS, BS):
            b = tr[perm[i:i + BS]].to(dev)
            for p in range(K, SEQ):
                ctx = torch.stack([b[:, p - 1 - j] for j in range(K)], 1); y = b[:, p]
                feat, _ = fwd(ctx); lg = logits(feat); P = len(y)
                e = torch.softmax(lg, 1); e[torch.arange(P, device=dev), y] -= 1
                head -= lr * (feat.t() @ e) / P
                # feedback to embedding: random  OR  sign of direct readout path head[:d]
                B = B_rand if mode == "random" else torch.sign(head[:d]).t() * 0.1
                dsig = (e @ B) / P
                for j in range(K):
                    E.index_add_(0, ctx[:, j], -lr * dsig)

    @torch.no_grad()
    def reply(prompt, n=22, temp=0.4):
        idl = [w2i.get(w, 1) for w in ["<user>"] + prompt.lower().split() + ["<asst>"]]
        while len(idl) < K:
            idl = [w2i["<user>"]] + idl
        out = []
        for _ in range(n):
            ctx = torch.tensor([[idl[-1 - j] for j in range(K)]], device=dev)
            feat, _ = fwd(ctx); pp = torch.softmax(logits(feat)[0] / temp, 0).cpu()
            nx = int(torch.multinomial(pp, 1).item())
            if vocab[nx] in ("<user>", "<pad>", "<asst>"):
                break
            idl.append(nx); out.append(vocab[nx])
        return " ".join(out) if out else "(empty)"

    print(f"\n===== DFA feedback = {mode}  val_ppl={valppl():.1f} =====", flush=True)
    torch.manual_seed(1)
    for q in ["i am really sad", "my dog is sick", "i got promoted at work", "i feel lonely"]:
        print(f"USER: {q}\nBOT : {reply(q)}", flush=True)


print("(backprop baseline 96.8, random-DFA ~160)", flush=True)
run("random")
run("sign")
