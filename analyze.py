"""
0단계 성공기준 검증. 학습 후 실행: python analyze.py

기준:
  1. 비슷한 토큰 cos ↑ / 다른 토큰 cos ↓
  2. 단일 전역 attractor 붕괴 안 함 (평균 쌍별 cos 낮음, V1~4는 0.999)
  3. 구별되는 attractor 수 > 0.14·N (Hopfield 한계 초과)
"""
import torch
import torch.nn.functional as F

from data import load_wikitext2
from model import SparsePatternLM

VOCAB = 2000
N     = 4000

PAIRS = [
    ("cat",  "dog"),    # 의미 유사   → 높길 기대
    ("cat",  "king"),   # 무관       → 낮길 기대
    ("the",  "a"),      # 둘 다 관사  → 높길 기대
    ("said", "told"),   # 유의어     → 중상
    ("one",  "two"),    # 둘 다 숫자  → 중간
    ("good", "bad"),    # 반의어     → ?
]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def count_attractors(binary: torch.Tensor, thresh: float = 0.9) -> int:
    """cos > thresh면 같은 attractor로 묶어 군집 수 세기 (greedy)."""
    bn = F.normalize(binary + 1e-8, dim=1)
    assigned = torch.full((bn.shape[0],), -1, dtype=torch.long)
    clusters = 0
    for i in range(bn.shape[0]):
        if assigned[i] >= 0:
            continue
        sims = bn @ bn[i]
        members = (sims > thresh) & (assigned < 0)
        assigned[members] = clusters
        clusters += 1
    return clusters


def main() -> None:
    _, _, _, w2i = load_wikitext2(max_vocab=VOCAB, seq_len=32, batch_size=32)

    model = SparsePatternLM(vocab_size=VOCAB, N=N)
    model.load_state_dict(torch.load("model_best.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        all_patterns = model.encode(torch.arange(VOCAB))   # (V, N)
        binary = (all_patterns > 0.5).float()

    # 1. 토큰 쌍 유사도
    print(f"\n{'word1':10s}  {'word2':10s}  cos / overlap")
    print("-" * 42)
    for w1, w2 in PAIRS:
        if w1 not in w2i or w2 not in w2i:
            print(f"{w1!r}/{w2!r} not in vocab — skip")
            continue
        b1, b2 = binary[w2i[w1]], binary[w2i[w2]]
        sim = cosine(b1, b2)
        overlap = (b1 * b2).sum().item()
        kmin = min(b1.sum().item(), b2.sum().item())
        print(f"{w1:10s}  {w2:10s}  cos={sim:+.3f}  overlap={overlap:.0f}/{kmin:.0f}")

    # 2. 붕괴 체크 — 평균 쌍별 cos
    bn = F.normalize(binary + 1e-8, dim=1)
    cos = bn @ bn.t()
    off_mean = (cos.sum() - cos.diagonal().sum()) / (VOCAB * (VOCAB - 1))
    k_mean = binary.sum(dim=1).mean().item()

    # 3. attractor 수
    n_attr = count_attractors(binary)
    hopfield = 0.14 * N

    print("\n--- 성공기준 ---")
    print(f"평균 쌍별 cos     : {off_mean:.3f}   (낮을수록 좋음; V1~4=0.999=붕괴)")
    print(f"평균 활성 수 k    : {k_mean:.0f} / {N}")
    print(f"구별 attractor 수 : {n_attr}   (Hopfield 0.14N={hopfield:.0f}, vocab={VOCAB})")
    print(f"판정: 붕괴 {'없음' if off_mean < 0.5 else '발생'} | "
          f"용량 {'통과' if n_attr > hopfield else '미달'}")

    # top-5 유사 단어
    query = "king"
    if query in w2i:
        print(f"\n'{query}' 최근접 5개:")
        sims = F.cosine_similarity(bn[w2i[query]].unsqueeze(0), bn)
        top = sims.topk(6)
        vocab_list = list(w2i.keys())
        for val, idx in zip(top.values[1:], top.indices[1:]):
            print(f"  {vocab_list[idx]:15s}  {val.item():+.3f}")


if __name__ == "__main__":
    main()
