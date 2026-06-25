"""
0단계 v3 학습 — 순수 STDP (gradient 없음).
시퀀스의 연속 토큰 쌍을 STDP로 묶음: assembly(t) → assembly(t+1).
예측: seed c → settle → 전이 → 다음 assembly와 overlap.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_wikitext2
from model import SparsePatternLM

VOCAB  = 2000
N      = 4000
EPOCHS = 3
BS     = 32
SEQ    = 16


@torch.no_grad()
def pattern_stats(model: SparsePatternLM):
    """assembly 평균 쌍별 cos(붕괴 감시), 평균 k, 전이 이탈도."""
    asm = model.all_assemblies()                       # (V,N) soft
    b = (asm > 0.5).float()
    k = b.sum(1).mean().item()
    bn = F.normalize(b + 1e-8, dim=1)
    cos = bn @ bn.t()
    V = cos.shape[0]
    off = ((cos.sum() - cos.diagonal().sum()) / (V * (V - 1))).item()
    # 전이 이탈: predict(assembly_c)가 자기 c에서 얼마나 벗어나나 (낮으면 갇힘)
    pred = model.predict(asm)
    pb = F.normalize((pred > 0.5).float() + 1e-8, dim=1)
    self_cos = (pb * bn).sum(1).mean().item()
    return off, k, self_cos


@torch.no_grad()
def evaluate(model, val_batches, device):
    """예측 overlap → CE. assembly 캐시 1회."""
    asm = model.all_assemblies()                       # (V,N)
    total, n = 0.0, 0
    for inp, tgt in val_batches:
        inp, tgt = inp.to(device), tgt.to(device)
        B, T = inp.shape
        a = asm[inp.reshape(-1)]                        # (B*T, N) context assembly
        pred = model.predict(a)                         # 전이
        logits = pred @ asm.t()                         # (B*T, V) overlap
        total += F.cross_entropy(logits, tgt.reshape(-1), ignore_index=0).item()
        n += 1
    return total / n


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_batches, val_batches, vocab, _ = load_wikitext2(max_vocab=VOCAB, seq_len=SEQ, batch_size=BS)
    print(f"train={len(train_batches)} val={len(val_batches)} batches")

    model = SparsePatternLM(vocab_size=VOCAB, N=N).to(device)
    print(f"N={N} vocab={VOCAB} k0={model.k0}")

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        for inp, _tgt in tqdm(train_batches, desc=f"epoch {epoch}", leave=False):
            inp = inp.to(device)
            B, T = inp.shape
            prev = None
            for t in range(T):
                a = model.assembly_of(inp[:, t])       # (B,N) assembly_t
                if prev is not None:
                    model.stdp(prev, a)                 # prev(t-1) → a(t)
                model.homeostasis(a)
                prev = a

        vl = evaluate(model, val_batches, device)
        off, k, self_cos = pattern_stats(model)
        nz = (model.W.abs() > 1e-4).float().mean().item()
        print(f"epoch {epoch} | val={vl:.3f} ppl={2.718**vl:.1f} | "
              f"W_nz={nz:.3f} | asm_cos={off:.3f} | k={k:.0f} | self_cos={self_cos:.3f}")
        if vl < best:
            best = vl
            torch.save({"W": model.W, "theta": model.theta, "anchor": model.anchor}, "model_best.pt")

    print(f"best val={best:.3f} ppl={2.718**best:.1f}")


if __name__ == "__main__":
    main()
