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
    def __init__(self, mode, n=1, beta=0.5, iters=5, dt=1.0):
        super().__init__()
        self.mode = mode
        self.n = n  # divisive-norm exponent (Heeger 1992 / Carandini&Heeger 2012 canonical form has n~2,
                    # not n=1 -- prior divnorm run used plain ratio (n=1), missing the expansive
                    # nonlinearity that's the actual bio mechanism for sharp competition, not softmax-mimicry.
        self.beta = beta    # lateral inhibition strength (Grossberg shunting on-center off-surround)
        self.iters = iters  # recurrent settling steps (real circuit doesn't normalize in one shot)
        self.dt = dt        # euler step size for leaky integration; dt=1.0 = original (no decay, full
                             # replace each step); dt<1 = partial step w/ leak (-x term), closer to real
                             # continuous shunting dynamics, may avoid overshoot/oscillation.
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
        elif mode in ("phase", "phase2", "phase_pos"):
            # theta-gamma / binding-by-synchrony analogue (Fries 2005 communication-through-coherence;
            # von der Malsburg binding-by-synchrony) -- NOT a normalization mechanism, unlike every
            # other non-add mode above (softmax/divnorm/lateral all divide by a window-wide sum).
            # phase: each token's coupling strength is gated purely by ITS OWN content-match phase via
            # Malus's law (cos^2), no cross-token normalization: w_j = cos(phi_j)^2, phi_j in [0,pi/2]
            # set by (query,key_j) match. High match -> phi->0 (in-phase, strong coupling); low match
            # -> phi->pi/2 (out-of-phase, decoupled). mix = sum_j w_j*v_j, NOT divided by sum(w) --
            # amplitude summation of coherent oscillators, not a competitive ratio.
            # phase2: extends this to real population coupling (window tokens' phases pull each other
            # via Kuramoto coupling, not just independently set by query match) -- real binding-by-
            # synchrony has mutually-entraining oscillators, not per-token-independent gating.
            # phase_pos: order_probe.py found phase/phase2 are BOTH order-blind (no positional signal
            # anywhere in q/k/v). real theta-gamma serial-order coding (Lisman & Idiart 1995; Jensen &
            # Lisman 1996 working-memory model) nests each item's spike volley in a distinct gamma
            # sub-cycle WITHIN the theta cycle by serial position -- phase encodes "when", not (as in
            # "phase" above) "how well it matches". here: fixed per-slot phase offset pos_j=(pi/2)*j/(K-1)
            # (j=0 is "now"/cur, j=K-1 is oldest) is ADDED to the content-match phase, so a token only
            # gets high coherence gain if it matches AND arrives near the expected serial slot -- true
            # binding-by-synchrony binds "what" (content phase) to "when" (position phase) in one gate.
            self.Wq = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wv = nn.Linear(d, d, bias=False)
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
            # improvement: drive raised to n before settling -- divnorm ablation showed the n~2
            # expansive nonlinearity (not the ratio/settling shape) is what recovers most of the gap,
            # so combine both bio ingredients: expansive drive + recurrent shunting settling.
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            raw = F.relu((q * k).sum(-1) / math.sqrt(d)) ** self.n   # (B,K) expansive drive
            x = raw.clone()
            for _ in range(self.iters):
                S = x.sum(1, keepdim=True)
                target = F.relu(raw - self.beta * (S - x))
                x = x + self.dt * (target - x)   # dt=1 -> x=target (original); dt<1 -> leaky partial step
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
        elif self.mode == "matmul":  # low-rank matrix transition, cross-dim mixing via bottleneck
            h = torch.zeros_like(cur)
            for j in reversed(range(K)):
                tok = emb[:, j]
                g = torch.sigmoid(self.Bg(tok))
                hb = self.C(h)
                h = self.A(g * hb) + self.Win(tok)
            mix = h
        elif self.mode == "phase":  # binding-by-synchrony, coherence-gated (NOT normalized) amp. sum
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            s = (q * k).sum(-1) / math.sqrt(d)                    # (B,K) content-match score
            phi = (math.pi / 2) * (1 - torch.sigmoid(self.beta * s))  # high match -> phi->0 (in-phase)
            w = torch.cos(phi) ** 2                               # Malus's law coherence gain, per-token
            mix = (w.unsqueeze(-1) * v).sum(1)                    # no sum(w) normalization
        elif self.mode == "phase2":  # window-wide mutual coherence (Kuramoto population coupling), not
            # per-token-independent gating. tokens with similar content pull each other's phase together
            # (real assembly formation), then the settled phase (relative to query, phase=0 reference)
            # sets the coherence gain -- same Malus's law readout as "phase", different phase *dynamics*.
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            s = (q * k).sum(-1) / math.sqrt(d)                          # (B,K) init match to query
            phi = (math.pi / 2) * (1 - torch.sigmoid(self.beta * s))    # init phase, as in "phase"
            coup = torch.tanh(torch.einsum("bid,bjd->bij", k, k) / math.sqrt(d))  # (B,K,K) mutual coupling
            for _ in range(self.iters):
                # Kuramoto: dphi_i = mean_j coup_ij * sin(phi_j - phi_i); self-term (i=j) vanishes (sin(0)=0)
                dphi = (coup * torch.sin(phi.unsqueeze(1) - phi.unsqueeze(2))).mean(-1)
                phi = phi + self.dt * dphi
            w = torch.cos(phi) ** 2                                     # coherence to query, post-settling
            mix = (w.unsqueeze(-1) * v).sum(1)
        else:  # phase_pos: theta-gamma serial-order code -- fixed per-slot phase offset (j=0 "now" ...
            # j=K-1 oldest) ADDED to content-match phase. binds "what" (content) to "when" (serial
            # position) in one coherence gate, unlike "phase"/"phase2" which are order-blind (order_probe.py).
            q = self.Wq(cur).unsqueeze(1)
            k = self.Wk(emb)
            v = self.Wv(emb)
            s = (q * k).sum(-1) / math.sqrt(d)
            phi_content = (math.pi / 2) * (1 - torch.sigmoid(self.beta * s))
            pos = torch.arange(K, device=emb.device, dtype=emb.dtype) * (math.pi / 2) / (K - 1)  # (K,)
            phi = phi_content + pos.unsqueeze(0)                        # (B,K), broadcast position offset
            w = torch.cos(phi) ** 2
            mix = (w.unsqueeze(-1) * v).sum(1)
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


def run(mode, n=1, beta=0.5, iters=5, dt=1.0):
    torch.manual_seed(0)
    m = Combiner(mode, n=n, beta=beta, iters=iters, dt=dt).to(dev)
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
        print(f"{mode:8s} n={n} beta={beta} it={iters} dt={dt} ep{ep:2d} ppl={ppl:.1f}", flush=True)
    print(f"  -> {mode} n={n} beta={beta} it={iters} dt={dt} params={npar/1e6:.2f}M BEST={best:.1f}", flush=True)
    return best


if __name__ == "__main__":
    # NB: experiment-runner code lives behind this guard so other scripts can safely
    # `from multiplicative_gate import Combiner` without re-triggering a full training run
    # (bit us once this session -- an import for a shape smoke-test accidentally launched pass9 for real).
    print("baselines: additive 61.8 | softmax-attn 58.6 | frozen-attn 56.8 (prior sessions)", flush=True)
    print("this run 2026-07-06 pass1: add 62.1 | gate(diag self) 60.4 | attn 57.4", flush=True)
    print("this run 2026-07-06 pass2: cmpgate 60.2 | matmul 60.5 (non-commutative alone doesn't close gap)", flush=True)
    print("this run 2026-07-06 pass3: divnorm(n=1) 59.7, divnorm(n=2) 59.4 best, divnorm(n=4) 59.8", flush=True)
    print("pass5: recurrent lateral inhibition (Grossberg shunting, settles over iters, not 1-shot div)", flush=True)
    print("prior interrupted run: lateral n=1 beta=0.3 = 61.8 (worse than divnorm 59.4)", flush=True)
    print("pass6 (dt=1, one-shot replace): lateral_n1_b0.7=62.7, lateral_n2_b0.3=61.2 best, lateral_n2_b0.7=62.7", flush=True)
    print("-> still below divnorm 59.4. pass7: euler leaky integration (dt<1, partial step w/ decay)", flush=True)
    print("   on best config (n=2, beta=0.3) to test if gradual settling beats one-shot replace", flush=True)
    print("pass7 result: dt=0.3->59.8(best) dt=0.5->62.1 dt=0.7->61.6 dt=1.0->60.9 -- ALL below 59.4.", flush=True)
    print("   normalization family (gate/cmpgate/matmul/divnorm/lateral) ceiling confirmed ~59.4.", flush=True)
    print("pass8: phase mode -- binding-by-synchrony (Fries communication-through-coherence), NOT a", flush=True)
    print("   normalization mechanism -- coherence gain w_j=cos(phi_j)^2 has no window-wide sum, unlike", flush=True)
    print("   every mode above. beta = match->phase steepness, sweeping to find working regime.", flush=True)
    print("pass8 result (already run, not re-run here): beta=0.5->58.5 beta=1.0->58.9 beta=2.0->58.4(best)", flush=True)
    print("   beta=4.0->62.0(too steep)", flush=True)
    print("-> phase BEATS divnorm 59.4 for the first time (normalization-family ceiling). gap to attn", flush=True)
    print("   57.4 narrows from 2.0p to 1.0p. but phase gates each token INDEPENDENTLY off query match --", flush=True)
    print("   no real population coupling yet. pass9: phase2 = Kuramoto mutual coupling among window", flush=True)
    print("   tokens (similar-content tokens pull each other's phase together = real assembly formation),", flush=True)
    print("   settled phase vs query sets final coherence gain. beta=2.0 fixed (pass8 winner), sweep iters.", flush=True)
    results2 = {}
    for iters in (1, 3, 5, 8):
        results2[f"phase2_b2.0_it{iters}"] = run("phase2", beta=2.0, iters=iters)
    print("=== SUMMARY (pass9) ===", flush=True)
    for mode, ppl in results2.items():
        print(f"{mode:14s} BEST={ppl:.1f}", flush=True)
    print("pass9 result: it=1->58.8 it=3->59.0 it=5->58.8 it=8->58.9 -- flat, NO improvement over phase", flush=True)
    print("   alone (58.4). order_probe.py (structural, no training) found attn/divnorm/lateral/phase/", flush=True)
    print("   phase2 ALL order-blind (window memory-slot shuffle -> Delta logit = 0.000000 exactly) --", flush=True)
    print("   only add/gate/cmpgate/matmul see order. pass10: phase_pos -- theta-gamma SERIAL-ORDER code", flush=True)
    print("   (Lisman-Idiart), fixed per-slot phase offset pos_j=(pi/2)*j/(K-1) ADDED to content-match", flush=True)
    print("   phase, so gate = f(what AND when) not just f(what). confirmed order-sensitive structurally", flush=True)
    print("   (order_probe: Delta=0.58). now test if it actually helps ppl.", flush=True)
    results3 = {}
    for beta in (1.0, 2.0, 4.0):
        results3[f"phase_pos_b{beta}"] = run("phase_pos", beta=beta)
    print("=== SUMMARY (pass10) ===", flush=True)
    for mode, ppl in results3.items():
        print(f"{mode:14s} BEST={ppl:.1f}", flush=True)
