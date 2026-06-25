import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePatternLMv4(nn.Module):
    """
    0단계 v4: 지속 상태(working memory) + 점화 + STDP. gradient 없음.

    v3 대비 변경:
      - 상태 리셋 X. leaky 적분(τ)으로 시퀀스 내내 persist → 문맥 누적.
        h ← (1-λ)·h + λ·target   (λ = 1/τ, 망각률)
      - E/I 완화 (θ·γ 낮춤) → 재귀 흥분이 점화 가능 (non-empty 유지)
      - 토큰 = 지속 상태에 anchor 주입 (리셋 아님)

    뇌 대응:
      지속 상태 = 전전두엽 working memory (sustained recurrent activity)
      leaky τ   = 막 시정수 (망각/적분)
      점화      = 재귀 흥분이 다음 assembly 켬 (pattern completion)
      STDP      = 문맥상태(t) → (t+1) 방향 결합
    """

    def __init__(
        self,
        vocab_size: int,
        N: int = 4000,
        k0: int = 10,
        lam: float = 0.5,          # leaky 적분률 (작을수록 긴 기억). τ≈1/λ
        n_settle: int = 4,         # 토큰당 적분 sub-step
        beta: float = 4.0,
        sigma: float = 0.5,
        gamma: float = 3.0,        # 피드백 억제 (v3 8→3, 점화 위해 완화)
        rho: float = 0.05,
        theta_init: float = 0.2,   # 발화 역치 (v3 0.5→0.2, 점화 위해 완화)
        eta_homeo: float = 0.02,
        seed_gain: float = 3.0,
        stdp_lr: float = 0.02,
        decay: float = 0.999,
        rec_gain: float = 1.0,     # 재귀 흥분 이득 (점화 세기)
    ):
        super().__init__()
        assert N > vocab_size
        self.vocab_size = vocab_size
        self.N = N
        self.k0 = k0
        self.lam = lam
        self.n_settle = n_settle
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.eta_homeo = eta_homeo
        self.seed_gain = seed_gain
        self.stdp_lr = stdp_lr
        self.decay = decay
        self.rec_gain = rec_gain

        anchor = torch.zeros(vocab_size, N)
        for v in range(vocab_size):
            anchor[v, torch.randperm(N)[:k0]] = 1.0
        self.register_buffer("anchor", anchor)

        W = torch.zeros(N, N)
        mask = torch.rand(N, N) < 0.02
        mask.fill_diagonal_(False)
        W[mask] = torch.randn(int(mask.sum().item())) * 0.1
        self.register_buffer("W", W)

        self.register_buffer("theta", torch.full((N,), float(theta_init)))

    def new_state(self, B: int, device) -> torch.Tensor:
        return torch.zeros(B, self.N, device=device)

    def _step(self, h: torch.Tensor, drive: torch.Tensor | None) -> torch.Tensor:
        """leaky 적분 한 스텝. drive=None이면 입력 없는 전이(예측)."""
        rec = self.rec_gain * (h @ self.W)
        total = rec if drive is None else rec + self.seed_gain * drive
        activity = h.mean(dim=1, keepdim=True)
        R = total / (self.sigma + self.gamma * activity)
        target = torch.sigmoid(self.beta * (R - self.theta.clamp(min=0.0)))
        return (1.0 - self.lam) * h + self.lam * target          # 지속(leaky)

    def integrate(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """현 상태에 토큰 anchor 주입하며 n_settle 적분. 상태 진화(리셋 X)."""
        drive = self.anchor[token_ids]
        for _ in range(self.n_settle):
            h = self._step(h, drive)
        return h

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """입력 없이 전이 → 다음 가리키는 상태."""
        s = h
        for _ in range(self.n_settle):
            s = self._step(s, None)
        return s

    def readout(self, pred: torch.Tensor) -> torch.Tensor:
        """다음 토큰 logits = 예측 상태와 anchor 겹침."""
        return pred @ self.anchor.t()                            # (B, V)

    @torch.no_grad()
    def stdp(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        B = pre.shape[0]
        dW = (pre.t() @ post) / B
        self.W.mul_(self.decay)
        self.W.add_(self.stdp_lr * dW)
        self.W.fill_diagonal_(0.0)

    @torch.no_grad()
    def homeostasis(self, h: torch.Tensor) -> None:
        rate = (h > 0.5).float().mean(dim=0)
        self.theta += self.eta_homeo * (rate - self.rho)
        self.theta.clamp_(min=0.0)
