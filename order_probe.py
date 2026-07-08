# PROBE: does each combiner mode actually use window ORDER (recency among the K-1 "memory"
# tokens), or is it a permutation-invariant bag-of-context (only position 0 = "cur" is special)?
# Structural property of the forward function -- untrained/random weights are enough to test it,
# no training loop needed. ctx[:,0]=t-1 (cur, used as query in attn-family modes) is kept fixed;
# positions 1..K-1 (t-2..t-K) are shuffled and we check if output logits change.
import torch
from multiplicative_gate import Combiner, K, VOCAB

torch.manual_seed(0)
B = 8
ctx = torch.randint(0, VOCAB, (B, K))
perm = torch.randperm(K - 1) + 1  # permutation of positions 1..K-1 only, position 0 untouched
ctx_perm = ctx.clone()
ctx_perm[:, 1:] = ctx[:, perm]

configs = [
    ("add", {}), ("gate", {}), ("cmpgate", {}), ("matmul", {}),
    ("attn", {}), ("divnorm", {"n": 2}), ("lateral", {"n": 2, "beta": 0.3}),
    ("phase", {"beta": 2.0}), ("phase2", {"beta": 2.0, "iters": 5}),
]

print(f"{'mode':10s} {'max|Δlogit|':>12s}  verdict")
for mode, kw in configs:
    m = Combiner(mode, **kw)
    m.eval()
    with torch.no_grad():
        out1 = m(ctx)
        out2 = m(ctx_perm)
    diff = (out1 - out2).abs().max().item()
    verdict = "ORDER-BLIND (bag over memory slots)" if diff < 1e-4 else "order-sensitive"
    print(f"{mode:10s} {diff:12.6f}  {verdict}")
