# GOAL: multiplicative token-gated recurrence vs additive-leaky vs softmax-attention.
#   gate: h_t = sigmoid(Wg(token_t)) * h_{t-1} + Win(token_t)   (dendritic shunting-inhibition
#         analogue: current token multiplicatively modulates gain of carried state, then injects
#         its own content). Elementwise multiply is order-sensitive across steps (unlike additive
#         leaky h<-(1-lam)h+lam*target, which is commutative -> proven binding-incapable, see
#         model_v13.py:19 / stdp_arc_state 2026-06-28 working-memory ablation, best=61 at lam=0,
#         monotonically worse as lam grows).
#   hypothesis: gate beats add(61.8), and closes toward attn(58.6) without softmax (more bio-local
#   than global normalization; shunting inhibition is a real local dendritic mechanism).
import torch, torch.nn as nn, torch.nn.functional as F, math
from data import load_wikitext2

dev = "mps"
VOCAB = 2000
d = 64
K = 8
EP = 12

tb, vb, vocab, w2i = load_wikitext2(max_vocab=VOCAB, seq_len=16, batch_size=32)


class Combiner(nn.Module):
    def __init__(self, mode, n=1, beta=0.5, iters=5):
        super().__init__()
        self.mode = mode
        self.n = n  # divisive-norm exponent (Heeger 1992 / Carandini&Heeger 2012 canonical form has n~2,
                    # not n=1 -- prior divnorm run used plain ratio (n=1), missing the expansive
                    # nonlinearity that's the actual bio mechanism for sharp competition, not softmax-mimicry.
        self.beta = beta    # lateral inhibition strength (Grossberg shunting on-center off-surround)
        self.iters = iters  # recurrent settling steps (real circuit doesn't normalize in one shot)
        self.E = nn.Embedding(VOCAB, d)
        if mode == "attn":
            self.Wq = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wv = nn.Linear(d, d, bias=False)
        elif mode == "add":
            self.U = nn.ParameterList([nn.Parameter(torch.randn(d, d) * 0.1) for _ in range(K)])
        elif mode == "gate":
            self.Wg = nn.Linear(d, d)                    # token -> multiplicative gate
            self.Win = nn.Linear(d, d, bias=False)        # token -> injected content
        elif mode == "cmpgate":
            # gate = compare(cur, past_token) instead of past_token alone -> pairwise, non-commutative
            self.Wq = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Win = nn.Linear(d, d, bias=False)
        elif mode == "divnorm":
            # same Wq/Wk/Wv as attn, but competitive normalization = divisive norm (Carandini & Heeger,
            # canonical cortical computation) instead of softmax. isolates: is softmax's *shape* needed,
            # or just window-wide competitive normalization (which divnorm also provides)?
            self.Wq = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wv = nn.Linear(d, d, bias=False)
        elif mode == "lateral":
            # recurrent lateral inhibition (Grossberg 1973 shunting on-center off-surround; Amari
            # competitive dynamics): x_i <- relu(raw_i - beta*(sum_j x_j - x_i)), iterated to settle.
            # a real WTA circuit doesn't normalize in one division -- it converges over several steps
            # of mutual inhibition. same Wq/Wk/Wv as divnorm, only the competition dynamics differ.
            self.Wq = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wv = nn.Linear(d, d, bias=False)
        elif mode == "matmul":
            # low-rank *matrix* transition (Sutskever multiplicative RNN): h <- A(gate(tok) * C(h)) + Win(tok)
            # gate mixes dims via bottleneck A/C, unlike "gate" mode's pure per-dim self-scaling.
            r = 16
            self.C = nn.Linear(d, r, bias=False)
            self.Bg = nn.Linear(d, r)
            self.A = nn.Linear(r, d, bias=False)
            self.Win = nn.Linear(d, d, bias=False)
        else:
            raise ValueError(mode)
        self.head = nn.Linear(2 * d, VOCAB, bias=False)

    def forward(self, ctx):  # ctx (B,K): ctx[:,0]=t-1 (most recent) ... ctx[:,K-1]=t-K
        emb = self.E(ctx)
        cur = emb[:, 0]
        if self.mode == "attn":
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            a = torch.softmax((q * k).sum(-1) / math.sqrt(d), dim=1)
            mix = (a.unsqueeze(-1) * v).sum(1)
        elif self.mode == "add":
            mix = sum(emb[:, j] @ self.U[j] for j in range(K))
        elif self.mode == "gate":  # recurrent multiplicative combination, oldest -> newest
            h = torch.zeros_like(cur)
            for j in reversed(range(K)):
                tok = emb[:, j]
                g = torch.sigmoid(self.Wg(tok))
                h = g * h + self.Win(tok)
            mix = h
        elif self.mode == "divnorm":  # canonical cortical normalization (Carandini & Heeger 2012):
            # R_i = L_i^n / (sigma^n + sum_j L_j^n). n=1 (plain ratio) is NOT the real model -- the
            # expansive power-law nonlinearity (n~2, measured in V1) is what produces sharp competition
            # in actual cortex, not an ad hoc softmax substitute.
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            raw = F.relu((q * k).sum(-1) / math.sqrt(d))          # (B,K) non-negative drive L_i
            p = raw ** self.n
            w = p / (p.sum(1, keepdim=True) + 1.0 ** self.n)       # sigma=1
            mix = (w.unsqueeze(-1) * v).sum(1)
        elif self.mode == "lateral":  # recurrent mutual inhibition settling, not one-shot division
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            raw = F.relu((q * k).sum(-1) / math.sqrt(d))     # (B,K) drive
            x = raw.clone()
            for _ in range(self.iters):
                S = x.sum(1, keepdim=True)
                x = F.relu(raw - self.beta * (S - x))
            w = x / (x.sum(1, keepdim=True) + 1.0)
            mix = (w.unsqueeze(-1) * v).sum(1)
        elif self.mode == "cmpgate":  # gate from comparing cur against each folded-in token
            q = self.Wq(cur)
            h = torch.zeros_like(cur)
            for j in reversed(range(K)):
                tok = emb[:, j]
                score = (q * self.Wk(tok)).sum(-1, keepdim=True) / math.sqrt(d)
                g = torch.sigmoid(score)
                h = g * h + self.Win(tok)
            mix = h
        else:  # matmul: low-rank matrix transition, cross-dim mixing via bottleneck
            h = torch.zeros_like(cur)
            for j in reversed(range(K)):
                tok = emb[:, j]
                g = torch.sigmoid(self.Bg(tok))
                hb = self.C(h)
                h = self.A(g * hb) + self.Win(tok)
            mix = h
        return self.head(torch.cat([cur, F.relu(mix)], -1))


def batches(split):
    out = []
    for inp, _ in split:
        T = inp.shape[1]
        for p in range(K, T):
            out.append((torch.stack([inp[:, p - 1 - j] for j in range(K)], 1), inp[:, p]))
    return out


trb = batches(tb)
vbb = batches(vb)


def run(mode, n=1, beta=0.5, iters=5):
    torch.manual_seed(0)
    m = Combiner(mode, n=n, beta=beta, iters=iters).to(dev)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best = 1e9
    for ep in range(1, EP + 1):
        m.train()
        for ctx, y in trb:
            ctx = ctx.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            F.cross_entropy(m(ctx), y).backward()
            opt.step()
        m.eval()
        tot = 0
        cnt = 0
        with torch.no_grad():
            for ctx, y in vbb:
                ctx = ctx.to(dev)
                y = y.to(dev)
                tot += F.cross_entropy(m(ctx), y, reduction="sum").item()
                cnt += len(y)
        ppl = math.exp(tot / cnt)
        best = min(best, ppl)
        print(f"{mode:8s} n={n} beta={beta} it={iters} ep{ep:2d} ppl={ppl:.1f}", flush=True)
    print(f"  -> {mode} n={n} beta={beta} it={iters} params={npar/1e6:.2f}M BEST={best:.1f}", flush=True)
    return best


print("baselines: additive 61.8 | softmax-attn 58.6 | frozen-attn 56.8 (prior sessions)", flush=True)
print("this run 2026-07-06 pass1: add 62.1 | gate(diag self) 60.4 | attn 57.4", flush=True)
print("this run 2026-07-06 pass2: cmpgate 60.2 | matmul 60.5 (non-commutative alone doesn't close gap)", flush=True)
print("this run 2026-07-06 pass3: divnorm(n=1) 59.7, divnorm(n=2) 59.4 best, divnorm(n=4) 59.8", flush=True)
print("pass5: recurrent lateral inhibition (Grossberg shunting, settles over iters, not 1-shot div)", flush=True)
results = {}
for beta in (0.3, 0.7):
    results[f"lateral_b{beta}"] = run("lateral", beta=beta, iters=5)
print("=== SUMMARY ===", flush=True)
for mode, ppl in results.items():
    print(f"{mode:14s} BEST={ppl:.1f}", flush=True)
