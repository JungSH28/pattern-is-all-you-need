"""
v13 학습/평가 — 이성=조합연산 (B안).
  학습: structured 정체성 코드 + Hebbian 카운트(bigram Wbi + 조합 Wco + unigram U). 1패스.
  측정: val ppl. 조합신호는 bigram 조밀 영역(작은 vocab)선 손해, 희소 영역(큰 vocab)서 이득.
실측 (vocab10000): 무작위코드 486 → 구조 bigram 311 → +조합 302.7.
"""
import torch
import torch.nn.functional as F

from data import load_wikitext2
from model_v13 import ReasoningLMv13

VOCAB = 10000
CFG = dict(n_in=4000, m=6000, k0=10, theta_conj=3.0, mix_w=0.3, alpha=0.01)


@torch.no_grad()
def evaluate(model, val_batches, device, n=200):
    tot, cnt = 0.0, 0
    for inp, _ in val_batches[:n]:
        inp = inp.to(device)
        for row in inp:
            logits = model.logits(row[:-2], row[1:-1])   # (t-1, t) → t+1
            tot += F.cross_entropy(logits, row[2:], ignore_index=0).item()
            cnt += 1
    return 2.718 ** (tot / cnt)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    train_batches, val_batches, vocab, w2i = load_wikitext2(
        max_vocab=VOCAB, seq_len=16, batch_size=32
    )
    print(f"train={len(train_batches)} val={len(val_batches)} cfg={CFG}")

    model = ReasoningLMv13(vocab_size=VOCAB, **CFG).to(device)
    model.fit(train_batches, device)

    # 조합 절제: mix_w=0 = 직접 bigram만
    base_w = model.mix_w
    model.mix_w = 0.0
    ppl_bi = evaluate(model, val_batches, device)
    model.mix_w = base_w
    ppl_conj = evaluate(model, val_batches, device)
    print(f"bigram only (w=0) ppl={ppl_bi:.1f} | +조합 (w={base_w}) ppl={ppl_conj:.1f}")

    torch.save({k: getattr(model, k) for k in
                ["P1", "P2", "code", "W_bi", "W_co", "U"]} | {"cfg": CFG},
               "model_v13_best.pt")


if __name__ == "__main__":
    main()
