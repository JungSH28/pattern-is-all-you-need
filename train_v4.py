"""
0단계 v4 학습 — 지속 상태(working memory) + STDP.
시퀀스 내내 상태 persist, 토큰마다 anchor 주입, 연속 문맥상태 쌍 STDP.
예측: 문맥상태 → 전이 → anchor overlap.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_wikitext2
from model_v4 import SparsePatternLMv4

VOCAB  = 2000
N      = 4000
EPOCHS = 3
BS     = 32
SEQ    = 16


@torch.no_grad()
def evaluate(model, val_batches, device):
    total, n = 0.0, 0
    act_sum, act_n = 0.0, 0
    for inp, tgt in val_batches:
        inp, tgt = inp.to(device), tgt.to(device)
        B, T = inp.shape
        h = model.new_state(B, device)
        loss = 0.0
        for t in range(T - 1):
            h = model.integrate(h, inp[:, t])
            pred = model.predict(h)
            logits = model.readout(pred)
            loss += F.cross_entropy(logits, inp[:, t + 1], ignore_index=0)
            act_sum += (pred > 0.5).float().sum(1).mean().item(); act_n += 1
        total += (loss / (T - 1)).item(); n += 1
    return total / n, act_sum / max(act_n, 1)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train_batches, val_batches, vocab, _ = load_wikitext2(max_vocab=VOCAB, seq_len=SEQ, batch_size=BS)
    print(f"train={len(train_batches)} val={len(val_batches)} batches")

    model = SparsePatternLMv4(vocab_size=VOCAB, N=N).to(device)
    print(f"N={N} vocab={VOCAB} k0={model.k0} lam={model.lam}")

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        for inp, _ in tqdm(train_batches, desc=f"epoch {epoch}", leave=False):
            inp = inp.to(device)
            B, T = inp.shape
            h = model.new_state(B, device)
            prev = None
            for t in range(T):
                h = model.integrate(h, inp[:, t])    # 문맥 누적
                if prev is not None:
                    model.stdp(prev, h)              # 문맥(t-1) → (t)
                model.homeostasis(h)
                prev = h.detach()

        vl, act = evaluate(model, val_batches, device)
        nz = (model.W.abs() > 1e-4).float().mean().item()
        print(f"epoch {epoch} | val={vl:.3f} ppl={2.718**vl:.1f} | W_nz={nz:.3f} | pred_active={act:.1f}")
        if vl < best:
            best = vl
            torch.save({"W": model.W, "theta": model.theta, "anchor": model.anchor}, "model_v4_best.pt")

    print(f"best val={best:.3f} ppl={2.718**best:.1f}")


if __name__ == "__main__":
    main()
