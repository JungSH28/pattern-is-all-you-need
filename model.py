import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePatternLM(nn.Module):
    """
    0단계 v3: flat + 시간 비대칭(STDP) 코어. gradient/backprop 없음.

    설계 (D1~D14):
      벡터 i = W[i,:] (단일 W). 토큰 = 수렴 assembly (k 유동).
      입력  : 고정 무작위 k₀-hot anchor (감각 안정성). 일시 입력전류 (영구 pin X).
      활성  : sigmoid. 흥분/억제 = W 부호.
      안정화: 항상성 역치 θ (intrinsic plasticity) + divisive 피드백억제 (GABA_A).
      학습  : W = STDP (pre(t)→post(t+1), 방향성·비대칭) + 감쇠(가지치기).
              θ = 항상성. k = 창발.
      예측  : seed c → settle → 한 스텝 전이(입력 없이) → next assembly와 overlap.

    공간 층 없음(시간 분리로 예측). Wernicke/Broca·궁상다발 = 2단계.
    """

    def __init__(
        self,
        vocab_size: int,
        N: int = 4000,
        k0: int = 10,              # anchor 크기 (고정, 작게 → 직교)
        n_settle: int = 3,         # 토큰 제시 시 수렴 sub-iter
        beta: float = 4.0,
        sigma: float = 0.5,
        gamma: float = 8.0,        # 피드백 억제 강도
        rho: float = 0.03,         # 목표 발화율 (언어피질 희소도)
        theta_init: float = 0.5,
        eta_homeo: float = 0.02,   # 항상성 적응 속도
        seed_gain: float = 4.0,    # anchor 입력전류 세기
        stdp_lr: float = 0.02,     # STDP 강화율
        decay: float = 0.999,      # W 감쇠 (가지치기)
        noise: float = 0.0,
    ):
        super().__init__()
        assert N > vocab_size
        self.vocab_size = vocab_size
        self.N = N
        self.k0 = k0
        self.n_settle = n_settle
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.eta_homeo = eta_homeo
        self.seed_gain = seed_gain
        self.stdp_lr = stdp_lr
        self.decay = decay
        self.noise = noise

        # 고정 무작위 k₀-hot anchor codebook (V, N) — 학습 안 함
        anchor = torch.zeros(vocab_size, N)
        for v in range(vocab_size):
            idx = torch.randperm(N)[:k0]
            anchor[v, idx] = 1.0
        self.register_buffer("anchor", anchor)

        # W: 방향성 비대칭. autograd 아님 (STDP로 수동 갱신). sparse 초기화.
        W = torch.zeros(N, N)
        mask = torch.rand(N, N) < 0.02
        mask.fill_diagonal_(False)
        W[mask] = torch.randn(int(mask.sum().item())) * 0.1
        self.register_buffer("W", W)

        # 항상성 역치 buffer
        self.register_buffer("theta", torch.full((N,), float(theta_init)))

    # ---- 동역학 ----

    def _step(self, state: torch.Tensor, input_drive: torch.Tensor | None) -> torch.Tensor:
        """한 스텝 갱신. input_drive=None이면 입력 없는 전이(예측)."""
        rec = state @ self.W                                    # 재귀 흥분/억제
        total = rec if input_drive is None else rec + self.seed_gain * input_drive
        activity = state.mean(dim=1, keepdim=True)              # 총활동
        R = total / (self.sigma + self.gamma * activity)        # divisive 억제
        signal = R - self.theta.clamp(min=0.0)
        if self.noise > 0:
            signal = signal + torch.randn_like(signal) * self.noise
        return torch.sigmoid(self.beta * signal)               # sigmoid 발화율

    def settle(self, input_drive: torch.Tensor) -> torch.Tensor:
        """anchor 입력전류로 assembly 수렴. 일시 입력(클램프 아님)."""
        state = torch.zeros_like(input_drive)
        for _ in range(self.n_settle):
            state = self._step(state, input_drive)
        return state

    def predict(self, assembly: torch.Tensor) -> torch.Tensor:
        """입력 없이 한 스텝 전이 → 다음 assembly 추정."""
        return self._step(assembly, None)

    def assembly_of(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.settle(self.anchor[token_ids])             # (B, N)

    def all_assemblies(self) -> torch.Tensor:
        ids = torch.arange(self.vocab_size, device=self.W.device)
        return self.settle(self.anchor[ids])                   # (V, N)

    # ---- 학습 (STDP, 비-gradient) ----

    @torch.no_grad()
    def stdp(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        """W[i,j] += lr·pre_i(t)·post_j(t+1). 방향성(비대칭). + 감쇠."""
        B = pre.shape[0]
        dW = (pre.t() @ post) / B                              # (N,N) i→j
        self.W.mul_(self.decay)
        self.W.add_(self.stdp_lr * dW)
        self.W.fill_diagonal_(0.0)

    @torch.no_grad()
    def homeostasis(self, state: torch.Tensor) -> None:
        rate = (state > 0.5).float().mean(dim=0)               # 노드별 발화율
        self.theta += self.eta_homeo * (rate - self.rho)
        self.theta.clamp_(min=0.0)
