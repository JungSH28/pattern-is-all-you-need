# Close the fully-bio gap: route DFA credit through BOTH paths (direct + attention-mix),
# instead of the crude "same signal to all positions" (which stalled at ~169 vs backprop 96.8).
#   The embedding contributes to the readout via (a) cur (direct) and (b) mix = Σ a_j·(E@Wv).
#   crude : dsig = e@B_E, added to every ctx position (ignores attention structure).
#   full  : dfeat = e@B_out; direct half -> cur; mix half -> context embeddings weighted by
#           attention a_j (routes credit to the tokens that were actually attended).
# Fully bio: fixed-random feedback (no weight transport), attention weights a_j are local activations.
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
    B_E = (torch.randn(V, d, generator=g) * 0.1).to(dev)        # crude DFA feedback
    B_out = (torch.randn(V, 2 * d, generator=g) * 0.1).to(dev)  # full-path DFA feedback
    B_v = (torch.randn(d, d, generator=g) * 0.1).to(dev)        # mix-path random feedback

    def fwd(ctx, keep=False):
        emb = E[ctx]; cur = emb[:, 0]
        q, k, v = cur @ Wq, emb @ Wk, emb @ Wv
        a = torch.softmax((q.unsqueeze(1) * k).sum(-1) / math.sqrt(d), 1)
        mix = (a.unsqueeze(-1) * v).sum(1)
        feat = torch.cat([cur, F.relu(mix)], 1)
        return (feat, feat @ head, a, mix) if keep else (feat, feat @ head)

    @torch.no_grad()
    def valppl():
        tot, c = 0.0, 0
        for i in range(0, len(va) - 64, 64):
            b = va[i:i + 64].to(dev)
            for p in range(K, SEQ):
                ctx = torch.stack([b[:, p - 1 - j] for j in range(K)], 1)
                _, lg = fwd(ctx)
                tot += F.cross_entropy(lg, b[:, p], reduction='sum').item(); c += b.size(0)
        return math.exp(tot / c)

    BS = 64
    for ep in range(EP):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr) - BS, BS):
            b = tr[perm[i:i + BS]].to(dev)
            for p in range(K, SEQ):
                ctx = torch.stack([b[:, p - 1 - j] for j in range(K)], 1); y = b[:, p]
                feat, lg, a, mix = fwd(ctx, keep=True); P = len(y)
                e = torch.softmax(lg, 1); e[torch.arange(P, device=dev), y] -= 1
                head -= lr * (feat.t() @ e) / P
                if mode == "crude":
                    dsig = (e @ B_E) / P
                    for j in range(K):
                        E.index_add_(0, ctx[:, j], -lr * dsig)
                else:  # full path
                    dfeat = e @ B_out                      # (P, 2d)
                    dcur = dfeat[:, :d]                     # direct path -> cur
                    dmix = dfeat[:, d:] * (mix > 0).float() # relu' on mix half
                    mixcred = dmix @ B_v                    # (P,d) mix-path random feedback
                    for j in range(K):
                        contrib = a[:, j:j+1] * mixcred     # weighted by attention to token j
                        if j == 0:
                            contrib = contrib + dcur        # position 0 also gets direct
                        E.index_add_(0, ctx[:, j], -lr * contrib / P)

    @torch.no_grad()
    def reply(prompt, n=22, temp=0.4):
        idl = [w2i.get(w, 1) for w in ["<user>"] + prompt.lower().split() + ["<asst>"]]
        while len(idl) < K:
            idl = [w2i["<user>"]] + idl
        out = []
        for _ in range(n):
            ctx = torch.tensor([[idl[-1 - j] for j in range(K)]], device=dev)
            _, lg = fwd(ctx); pp = torch.softmax(lg[0] / temp, 0).cpu()
            nx = int(torch.multinomial(pp, 1).item())
            if vocab[nx] in ("<user>", "<pad>", "<asst>"):
                break
            idl.append(nx); out.append(vocab[nx])
        return " ".join(out) if out else "(empty)"

    print(f"\n===== DFA = {mode}  val_ppl={valppl():.1f} =====", flush=True)
    torch.manual_seed(1)
    for q in ["i am really sad", "my dog is sick", "i got promoted at work", "i feel lonely"]:
        print(f"USER: {q}\nBOT : {reply(q)}", flush=True)


print("(backprop 96.8, crude-DFA ~169)", flush=True)
run("crude")
run("full")
