import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePatternLMv11(nn.Module):
    """
    0단계 v11: 최소 모델 — 순수 Hebbian 공동출현 + log(Weber-Fechner) readout.

    v3~v10 진단 결론: 우리가 "뇌부합"이라며 더한 정교 기전(STDP 비대칭·항상성·
        divisive·delta·도파민·working memory)이 전부 가장 기본인 공동출현 통계를
        부숴서 ppl 936(거의 uniform 2000)에 갇혔음. unigram 125·bigram 78에 한참 못 미침.
    핵심 두 뿌리:
        ① 선형 logit→softmax 미보정 = 포화 → 확신-틀림 → 폭발. (raw 카운트 softmax는
           포화한다. 뇌 readout은 로그 압축 = Weber-Fechner.)
        ② 무작위 anchor 겹침이 연합정보 뭉갬 (log(C)=58 vs anchor투영=352, 6배).

    v11 = 둘 다 정직하게:
        W = Σ anchor[a] ⊗ anchor[b]   (순수 Hebbian 공동출현, 양수 누적)
        logits = log(relu((pre@W)@anchorᵀ) + ε)   (로그 압축 + 정류)
    Hebbian(가장 기본 시냅스 학습) + Weber-Fechner(감각 로그응답) 둘 다 뇌부합.
    정교 기전 전부 제거 — 단순함이 핵심.

    효율: per-token outer 대신 bigram 카운트 C(V,V) → W = Aᵀ C A 한 번.
    """

    def __init__(self, vocab_size: int, N: int = 4000, k0: int = 20, eps: float = 1e-6):
        super().__init__()
        assert N > vocab_size
        self.vocab_size = vocab_size
        self.N = N
        self.k0 = k0
        self.eps = eps

        anchor = torch.zeros(vocab_size, N)
        for v in range(vocab_size):
            anchor[v, torch.randperm(N)[:k0]] = 1.0
        self.register_buffer("anchor", anchor)
        self.register_buffer("W", torch.zeros(N, N))

    @torch.no_grad()
    def fit(self, bigram_count: torch.Tensor) -> None:
        """W = Aᵀ C A. C[a,b] = train에서 (a→b) 공동출현 횟수."""
        A = self.anchor
        self.W.copy_(A.t() @ bigram_count.to(A.device) @ A)
        self.W.fill_diagonal_(0.0)

    def logits(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """직전 토큰 → 다음 토큰 로그압축 logits. (재귀·문맥 없음 = 최소.)"""
        drive = self.anchor[prev_ids] @ self.W                   # (B, N)
        raw = drive @ self.anchor.t()                            # (B, V) ≈ 공동출현 카운트
        return torch.log(F.relu(raw) + self.eps)                 # Weber-Fechner 로그압축
