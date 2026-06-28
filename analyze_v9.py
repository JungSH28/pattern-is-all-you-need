"""
v9 성공기준 검증. 학습 후 실행: python analyze_v9.py

v9에서 토큰 anchor는 고정 무작위(k0-hot) → 직교가 설계.
따라서 의미는 anchor 자체가 아니라 학습된 W(전이)에 창발해야 함.
의미 검증 = 각 토큰의 출력 drive(=anchor@W, '다음에 뭘 켜나')를 비교.
  비슷한 단어 → 비슷한 다음토큰 분포 → drive cos ↑.
"""
import torch
import torch.nn.functional as F

from data import load_wikitext2
from model_v9 import SparsePatternLMv9

VOCAB, N = 2000, 4000

PAIRS = [
    ("cat",  "dog"),    # 의미 유사   → 높길 기대
    ("cat",  "king"),   # 무관       → 낮길 기대
    ("the",  "a"),      # 둘 다 관사  → 높길 기대
    ("said", "told"),   # 유의어     → 중상
    ("one",  "two"),    # 둘 다 숫자  → 중간
    ("good", "bad"),    # 반의어     → ?
]


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    _, _, _, w2i = load_wikitext2(max_vocab=VOCAB, seq_len=32, batch_size=32)

    ckpt = torch.load("model_v9_best.pt", map_location="cpu")
    model = SparsePatternLMv9(vocab_size=VOCAB, N=N, **ckpt["cfg"])
    model.W.copy_(ckpt["W"])
    model.theta.copy_(ckpt["theta"])
    model.anchor.copy_(ckpt["anchor"])
    model.eval()

    with torch.no_grad():
        # 각 토큰: anchor 주입 → 출력 drive (다음토큰 예측 분포의 원천)
        drive = model.anchor @ model.W            # (V, N) 다음에 켤 노드
        logits = drive @ model.anchor.t()         # (V, V) 다음토큰 점수
    dn = F.normalize(drive + 1e-8, dim=1)
    ln = F.normalize(logits + 1e-8, dim=1)

    # 1. 토큰 쌍: drive 유사도 + 다음토큰분포 유사도
    print(f"\n{'word1':8s} {'word2':8s}  drive_cos / logit_cos")
    print("-" * 44)
    for w1, w2 in PAIRS:
        if w1 not in w2i or w2 not in w2i:
            print(f"{w1!r}/{w2!r} not in vocab — skip"); continue
        i, j = w2i[w1], w2i[w2]
        print(f"{w1:8s} {w2:8s}  drive={cosine(dn[i], dn[j]):+.3f}  "
              f"logit={cosine(ln[i], ln[j]):+.3f}")

    # 2. anchor 직교성 확인 (설계상 ~0이어야)
    an = F.normalize(model.anchor + 1e-8, dim=1)
    acos = an @ an.t()
    a_off = (acos.sum() - acos.diagonal().sum()) / (VOCAB * (VOCAB - 1))

    # 3. drive 평균 쌍별 cos (붕괴 체크 — 다 같으면 붕괴)
    d_off = (dn @ dn.t()).sum()
    d_off = (d_off - dn.shape[0]) / (VOCAB * (VOCAB - 1))

    print("\n--- 구조 ---")
    print(f"anchor 평균 쌍별 cos : {a_off:.3f}   (설계상 ~0, 직교)")
    print(f"drive  평균 쌍별 cos : {d_off:.3f}   (낮을수록 분화; 1=붕괴)")
    print(f"W_nz                 : {(model.W.abs()>1e-4).float().mean():.3f}")
    print(f"W_neg(억제 비율)     : {(model.W<0).float().mean():.3f}")

    # 4. 최근접 단어 (의미 창발 정성 확인)
    vocab_list = list(w2i.keys())
    for query in ("king", "two", "the"):
        if query not in w2i: continue
        sims = F.cosine_similarity(ln[w2i[query]].unsqueeze(0), ln)
        top = sims.topk(6)
        print(f"\n'{query}' 다음토큰분포 최근접 5:")
        for val, idx in zip(top.values[1:], top.indices[1:]):
            print(f"  {vocab_list[idx]:15s} {val.item():+.3f}")


if __name__ == "__main__":
    main()
