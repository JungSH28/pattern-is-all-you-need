"""Bio conjunctive coding WITHOUT credit assignment.
Mixed selectivity (Fusi) / random expansion (dentate gyrus):
  fixed random nonlinear projection of (t-1,t-2) -> conjunctive hidden units.
Hebbian readout (local, no backprop): W_out = sum_t phi_t (x) onehot(next_t).
Predict: logits = log(relu(phi @ W_out) + eps)  (Weber-Fechner log readout).
Tests: can fixed random conjunction + Hebbian beat bigram ceiling (63)?
"""
import math
import sys

import torch

from data import load_wikitext2

torch.manual_seed(0)
SEQ = 16
M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192  # expansion dim
THETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # sparsity threshold (quantile)
EPS = 1e-6


def triples(batches):
    """yield (t2, t1, t) consecutive within each seq row."""
    t2s, t1s, ts = [], [], []
    for xb, yb in batches:  # xb (B, SEQ) input, yb target = next
        # within row: positions 1..SEQ-1 have prev (xb[:,i-1]) and prevprev (xb[:,i-2])
        x = xb
        for i in range(2, x.size(1)):
            t2s.append(x[:, i - 2])
            t1s.append(x[:, i - 1])
            ts.append(x[:, i])
    return torch.cat(t2s), torch.cat(t1s), torch.cat(ts)


def main():
    tr, va, vocab, _ = load_wikitext2(max_vocab=2000, seq_len=SEQ, batch_size=32)
    V = len(vocab)
    # fixed random expansion: each hidden unit = random readout of (prev, prevprev) codes
    R1 = torch.randn(M, V)
    R2 = torch.randn(M, V)

    def phi(t1, t2):
        # (N, M): mixed selectivity, relu nonlinearity = conjunction
        pre = R1[:, t1].t() + R2[:, t2].t()  # (N, M)
        if THETA > 0:
            thr = torch.quantile(pre, THETA, dim=1, keepdim=True)
            pre = pre - thr
        return torch.relu(pre)

    tr_t2, tr_t1, tr_t = triples(tr)
    va_t2, va_t1, va_t = triples(va)
    print(f"M={M} theta={THETA} V={V} train_triples={len(tr_t)}", flush=True)

    # Hebbian readout: W_out (M, V) = sum phi (x) onehot(next)
    W = torch.zeros(M, V)
    bs = 4096
    for i in range(0, len(tr_t), bs):
        p = phi(tr_t1[i : i + bs], tr_t2[i : i + bs])  # (b, M)
        nxt = tr_t[i : i + bs]
        W.index_add_(1, nxt, p.t())  # W[:, nxt] += p.t()

    # eval
    tot, cnt = 0.0, 0
    logW = None
    for i in range(0, len(va_t), bs):
        p = phi(va_t1[i : i + bs], va_t2[i : i + bs])  # (b, M)
        scores = p @ W  # (b, V)
        logits = torch.log(torch.relu(scores) + EPS)
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        nxt = va_t[i : i + bs]
        tot += -logp[torch.arange(len(nxt)), nxt].sum().item()
        cnt += len(nxt)
    ppl = math.exp(tot / cnt) if tot / cnt < 700 else float("inf")
    print(f"conjunctive ppl={ppl:.0f}", flush=True)


if __name__ == "__main__":
    main()
