"""B: bio-local code learning via competitive learning (WTA + Hebbian).
NO backprop. Hidden units = learned prototypes of context (t-1,t-2).
  - winner = argmax_m  P[m].x   (lateral inhibition WTA = nonlinear -> conjunction)
  - update winner toward input: P[w] += eta (x - P[w])   (local Hebbian)
Then Hebbian readout: count next-token per winner cluster.
Predict: cluster's next-token distribution (log readout).
Tests: do LEARNED nonlinear codes beat random expansion (99) and bigram ceiling (63)?
"""
import math
import sys

import torch

from data import load_wikitext2

torch.manual_seed(0)
SEQ = 16
M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048  # n prototypes
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # WTA winners
ETA = 0.02
EPS = 1e-6


def triples(batches):
    t2s, t1s, ts = [], [], []
    for xb, _ in batches:
        for i in range(2, xb.size(1)):
            t2s.append(xb[:, i - 2]); t1s.append(xb[:, i - 1]); ts.append(xb[:, i])
    return torch.cat(t2s), torch.cat(t1s), torch.cat(ts)


def main():
    tr, va, vocab, _ = load_wikitext2(max_vocab=2000, seq_len=SEQ, batch_size=32)
    V = len(vocab)
    # prototype halves for t-1 and t-2 (context = concat of two onehots in 2V space)
    P1 = torch.randn(M, V) * 0.01
    P2 = torch.randn(M, V) * 0.01

    tr_t2, tr_t1, tr_t = triples(tr)
    va_t2, va_t1, va_t = triples(va)
    print(f"M={M} topk={TOPK} V={V} train={len(tr_t)}", flush=True)

    bs = 2048

    def winners(t1, t2, k):
        s = P1[:, t1].t() + P2[:, t2].t()  # (b, M) match score
        return s.topk(k, dim=1).indices  # (b, k)

    # phase 1: competitive learning (online, one pass)
    for i in range(0, len(tr_t), bs):
        t1 = tr_t1[i : i + bs]; t2 = tr_t2[i : i + bs]
        w = winners(t1, t2, 1).squeeze(1)  # (b,)
        # move winner prototype toward onehot input (local Hebbian)
        # P1[w] += eta (onehot(t1) - P1[w]); update per-feature efficiently
        P1[w] *= (1 - ETA)
        P1.index_put_((w, t1), P1[w, t1] + ETA, accumulate=False)
        P2[w] *= (1 - ETA)
        P2.index_put_((w, t2), P2[w, t2] + ETA, accumulate=False)

    # phase 2: Hebbian readout (freeze prototypes), count next per winner cluster
    W = torch.zeros(M, V)
    for i in range(0, len(tr_t), bs):
        t1 = tr_t1[i : i + bs]; t2 = tr_t2[i : i + bs]; nx = tr_t[i : i + bs]
        wk = winners(t1, t2, TOPK)  # (b, k)
        for j in range(TOPK):
            W.index_put_((wk[:, j], nx), torch.ones(len(nx)), accumulate=True)

    # eval
    tot, cnt = 0.0, 0
    for i in range(0, len(va_t), bs):
        t1 = va_t1[i : i + bs]; t2 = va_t2[i : i + bs]; nx = va_t[i : i + bs]
        wk = winners(t1, t2, TOPK)
        scores = torch.zeros(len(nx), V)
        for j in range(TOPK):
            scores += W[wk[:, j]]
        logits = torch.log(scores + EPS)
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        tot += -logp[torch.arange(len(nx)), nx].sum().item()
        cnt += len(nx)
    ppl = math.exp(tot / cnt) if tot / cnt < 700 else float("inf")
    print(f"competitive ppl={ppl:.0f}", flush=True)


if __name__ == "__main__":
    main()
