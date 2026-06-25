"""
Pure Hebbian training — no gradient.
W updated only by co-activation (fire together → wire together).
Sim seed: context tokens + target token fired simultaneously during training.
Inference: context only → pattern matching → next token.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_wikitext2

VOCAB       = 2000
N           = 2000
EPOCHS      = 10
BS          = 32
SEQ         = 32

HEBB_LR     = 5e-4
HEBB_DECAY  = 0.999
NOISE_SCALE = 0.02
N_ITERS     = 8
INIT_CONN   = 0.05


def make_W(N: int, connectivity: float, device: str) -> torch.Tensor:
    W = torch.zeros(N, N, device=device)
    mask = torch.rand(N, N, device=device) < connectivity
    mask.fill_diagonal_(False)
    W[mask] = torch.randn(int(mask.sum().item()), device=device) * 0.01
    return W


def make_threshold(N: int, device: str) -> torch.Tensor:
    return torch.zeros(N, device=device)


def converge(
    seed: torch.Tensor,      # (B, N) binary seed — pinned throughout
    W: torch.Tensor,
    threshold: torch.Tensor,
    n_iters: int,
    noise_scale: float,
    training: bool,
) -> torch.Tensor:
    """Run convergence from seed. Returns (B, N) binary final state."""
    state = seed.clone()
    for _ in range(n_iters):
        noise = torch.randn_like(state) * noise_scale if training else 0.0
        signal = state @ W - threshold + noise
        state = (signal > 0).float()     # hard binary — no sigmoid
        state = state * (1.0 - seed) + seed  # pin seeds
    return state


def hebbian_step(
    state: torch.Tensor,     # (B, N) binary
    W: torch.Tensor,
) -> None:
    """In-place Hebbian update: excitatory + inhibitory."""
    active   = (state > 0.5).float()   # (B, N)
    inactive = 1.0 - active
    n = active.shape[0]

    hebb_exc = (active.T @ active)   / n   # (N, N)
    hebb_inh = (active.T @ inactive) / n   # (N, N)

    W.mul_(HEBB_DECAY)
    W.add_(HEBB_LR * (hebb_exc - hebb_inh))
    W.fill_diagonal_(0)


def get_canonical(W: torch.Tensor, threshold: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Canonical binary patterns for all tokens (no gradient)."""
    N = W.shape[0]
    device = W.device
    ids = torch.arange(vocab_size, device=device)
    seed = torch.zeros(vocab_size, N, device=device)
    seed[ids, ids] = 1.0
    state = converge(seed, W, threshold, N_ITERS, 0.0, training=False)
    return (state > 0.5).float()   # (vocab_size, N)


def evaluate(
    val_batches: list,
    W: torch.Tensor,
    threshold: torch.Tensor,
    canonical: torch.Tensor,
    vocab_size: int,
) -> float:
    total_loss = 0.0
    device = W.device
    with torch.no_grad():
        for inp, tgt in val_batches:
            inp, tgt = inp.to(device), tgt.to(device)
            B, T = inp.shape
            N = W.shape[0]

            # Encode each token independently (no context for simplicity)
            seed = torch.zeros(B * T, N, device=device)
            flat = inp.reshape(-1)
            seed[torch.arange(B * T), flat] = 1.0

            state = converge(seed, W, threshold, N_ITERS, 0.0, training=False)
            logits = state @ canonical.T          # (B*T, vocab_size)
            loss = F.cross_entropy(logits, tgt.reshape(-1), ignore_index=0)
            total_loss += loss.item()
    return total_loss / len(val_batches)


def main() -> None:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}")

    print("loading data…")
    train_batches, val_batches, vocab, _ = load_wikitext2(
        max_vocab=VOCAB, seq_len=SEQ, batch_size=BS
    )
    print(f"train={len(train_batches)}  val={len(val_batches)}  batches")

    W         = make_W(N, INIT_CONN, device)
    threshold = make_threshold(N, device)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --- train (sim seed Hebbian) ---
        for inp, tgt in tqdm(train_batches, desc=f"epoch {epoch:2d}", leave=False):
            inp, tgt = inp.to(device), tgt.to(device)
            B, T = inp.shape

            # For each position t, seed context (0..t-1) + target (t) together
            for t in range(1, T):
                seed = torch.zeros(B, N, device=device)
                # context tokens 0..t-1
                for s in range(t):
                    seed[torch.arange(B), inp[:, s]] = 1.0
                # target token t (sim seed)
                seed[torch.arange(B), inp[:, t]] = 1.0

                state = converge(seed, W, threshold, N_ITERS, NOISE_SCALE, training=True)
                hebbian_step(state, W)

        # --- val ---
        canonical = get_canonical(W, threshold, VOCAB)
        vl = evaluate(val_batches, W, threshold, canonical, VOCAB)

        nonzero = (W.abs() > 1e-4).float().mean().item()
        print(f"epoch {epoch:2d} | val={vl:.3f} | ppl={2**vl:.1f} | W_nonzero={nonzero:.3f}")

        if vl < best_val:
            best_val = vl
            torch.save({"W": W, "threshold": threshold}, "model_hebbian_best.pt")

    print(f"best val loss: {best_val:.3f}  ppl={2**best_val:.1f}")


if __name__ == "__main__":
    main()
