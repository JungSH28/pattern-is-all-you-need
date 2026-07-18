"""Working memory: hold an assembly after the input is gone.

Every earlier goal is feedforward -- present an input, read an answer. Thought
first needs a state that outlasts its input. An item is shown for a couple of
steps, the input is removed, and the substrate runs on its own recurrence for a
blank delay. If the state still decodes to the item that was shown, and several
items are held at once without merging into one attractor, that is maintenance.

This re-opens v12, which stored the whole vocabulary as fixed-point attractors
and collapsed. Here only a few items are held, briefly, and the tools v12 did
not have -- per-neuron homeostasis against explosion, local contrast against a
single merged attractor -- keep the held assemblies apart. The bio main line is
untouched: this is a separate high-contrast regime, so #31/#32/#35 hold by
construction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spatial_connectome import SUBSTRATE, ConnectomeConfig, SpatialConnectome


def _connectome(seed: int, contrast_strength: float) -> SpatialConnectome:
    return SpatialConnectome(
        ConnectomeConfig(
            n_input=256,
            n_substrate=512,
            n_output=64,
            out_degree=192,
            topology="random",
            homeostatic_threshold=True,
            local_contrast=True,
            contrast_strength=contrast_strength,
            seed=seed,
        )
    )


@dataclass(frozen=True)
class MemoryModel:
    connectome: SpatialConnectome
    substrate: torch.Tensor
    patterns: dict[str, torch.Tensor]
    present: int


def train(
    seed: int,
    *,
    n_items: int = 6,
    present: int = 2,
    contrast_strength: float = 2.0,
    hold_rate: float = 1.2,
    recurrent: bool = True,
) -> MemoryModel:
    connectome = _connectome(seed, contrast_strength)
    items = [f"i{index}" for index in range(n_items)]
    connectome.register_vocabulary(items)
    connectome.reset_recurrent()
    substrate = connectome.region == SUBSTRATE

    patterns: dict[str, torch.Tensor] = {}
    for item in items:
        state = _present(connectome, item, present)
        patterns[item] = state[substrate].clone()
        if recurrent:
            connectome.hold_assembly(state, rate=hold_rate)
    return MemoryModel(connectome, substrate, patterns, present)


def _present(connectome: SpatialConnectome, item: str, present: int) -> torch.Tensor:
    state = torch.zeros(connectome.config.n_units)
    sensory = torch.zeros(connectome.config.n_units)
    sensory[connectome.input_assemblies[item]] = 1.0
    for _ in range(present):
        state = connectome.step(state, sensory=sensory)
    return state


def recall(model: MemoryModel, item: str, blank: int) -> torch.Tensor:
    """Present the item, remove the input, run the blank delay on recurrence."""
    state = _present(model.connectome, item, model.present)
    for _ in range(blank):
        state = model.connectome.step(state, sensory=None)
    return state[model.substrate]


def decode(model: MemoryModel, held: torch.Tensor) -> str:
    similarities = {
        item: float(F.cosine_similarity(held[None], pattern[None]))
        for item, pattern in model.patterns.items()
    }
    return max(similarities, key=similarities.get)


def result(seed: int, *, blank: int, recurrent: bool = True) -> dict[str, object]:
    model = train(seed, recurrent=recurrent)
    items = list(model.patterns)
    held = {item: recall(model, item, blank) for item in items}

    correct = sum(decode(model, held[item]) == item for item in items)
    # Collapse check: the held states must stay apart, not merge into one point.
    cross = [
        float(F.cosine_similarity(held[a][None], held[b][None]))
        for index, a in enumerate(items)
        for b in items[index + 1 :]
    ]
    density = sum(float((held[item] > 0).float().mean()) for item in items) / len(items)
    return {
        "seed": seed,
        "correct": correct,
        "total": len(items),
        "mean_cross_cosine": sum(cross) / len(cross),
        "max_cross_cosine": max(cross),
        "density": density,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--blanks", type=int, nargs="+", default=[3, 5, 8])
    arguments = parser.parse_args()

    print(f"\n=== working memory, {arguments.seeds} seeds ===")
    print(f"  {'blank':>5} {'recurrent':>18} {'no-recurrent (ctrl)':>20} {'cross-cos':>10} {'density':>8}")
    for blank in arguments.blanks:
        rec = [result(seed, blank=blank, recurrent=True) for seed in range(arguments.seeds)]
        ctl = [result(seed, blank=blank, recurrent=False) for seed in range(arguments.seeds)]
        rc = sum(int(r["correct"]) for r in rec)
        rt = sum(int(r["total"]) for r in rec)
        cc = sum(int(r["correct"]) for r in ctl)
        ct = sum(int(r["total"]) for r in ctl)
        cross = sum(float(r["max_cross_cosine"]) for r in rec) / len(rec)
        density = sum(float(r["density"]) for r in rec) / len(rec)
        print(
            f"  {blank:5d} {rc:8d}/{rt:<8d} {cc:9d}/{ct:<9d} {cross:10.2f} {density:8.2f}"
        )
    print("  (control = developmental recurrence never strengthened; chance is 1/6)")


if __name__ == "__main__":
    main()
