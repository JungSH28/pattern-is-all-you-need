"""Multi-seed context-branch and spatial-topology ablation.

Required task:
    A, B -> C
    D, B -> E

The last input is identical. Success requires the storage/reasoning hot state to
retain the earlier context and the single connectome to select different output
assemblies. Three matched conditions isolate geometry:

* random: sparse edges ignore 3-D distance;
* distance: distance biases the same number of edges;
* developed: distance topology after early activity-dependent position motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from statistics import mean

import torch

from spatial_connectome import SUBSTRATE, ConnectomeConfig, SpatialConnectome


TOKENS = ("A", "B", "C", "D", "E", "F", "G")
BRANCHES = (("A", "B", "C"), ("D", "B", "E"))


@dataclass(frozen=True)
class ProbeResult:
    condition: str
    seed: int
    branch_correct: int
    reversed_rejected: int
    retained_after_new: int
    context_cosine: float
    edge_distance: float

    @property
    def success(self) -> bool:
        # The active goal is simultaneous context branching. Reversal and
        # later-memory retention remain reported diagnostics, but are separate
        # future capabilities and do not redefine this goal.
        return self.branch_correct == 2 and self.context_cosine < 0.95


def make_model(condition: str, seed: int) -> SpatialConnectome:
    topology = "random" if condition == "random" else "distance"
    model = SpatialConnectome(
        ConnectomeConfig(
            n_input=48,
            n_substrate=160,
            n_output=48,
            out_degree=40,
            topology=topology,
            # Two internal steps preserve the first token's transient while
            # still allowing B to recruit a context-specific conjunction.
            # The prior five-step setting settled too far toward a B-dominated
            # state (R cosine 0.92~0.97 across branches).
            steps_per_token=2,
            max_region_density=0.12,
            seed=seed,
        )
    )
    # F/G are deliberately novel at the later-learning audit; registering them
    # after development also prevents an unused future vocabulary item from
    # changing the developmental topology RNG stream.
    model.register_vocabulary(TOKENS[:5])
    if condition == "developed":
        traces = []
        # Development sees recurring sensory contexts, not target labels.
        for _ in range(4):
            for sequence in (("A", "B"), ("D", "B"), ("B", "A"), ("B", "D")):
                _, trace = model.run_sequence(sequence, keep_trace=True)
                traces.extend(trace)
        model.develop_positions(torch.stack(traces), epochs=3)
    model.register_vocabulary(TOKENS[5:])
    return model


def substrate_state(model: SpatialConnectome, context: tuple[str, str]) -> torch.Tensor:
    state = model.run_sequence(context, reset=True)
    assert isinstance(state, torch.Tensor)
    return state[model.region == SUBSTRATE]


def prediction(model: SpatialConnectome, context: tuple[str, str]) -> str:
    state = model.run_sequence(context, reset=True)
    assert isinstance(state, torch.Tensor)
    scores = model.output_scores(state, TOKENS)
    return max(scores, key=scores.get)


def train_branches(model: SpatialConnectome, rounds: int) -> None:
    for iteration in range(rounds):
        # Reverse order each round so neither target consistently receives the
        # last update.
        order = BRANCHES if iteration % 2 == 0 else tuple(reversed(BRANCHES))
        for first, last, target in order:
            model.learn_association(
                (first, last), target, learning_rate=0.08, target_steps=3
            )
        if iteration % 10 == 0:
            model.consolidate()


def evaluate(condition: str, seed: int, rounds: int = 220) -> ProbeResult:
    model = make_model(condition, seed)
    train_branches(model, rounds)

    branch_correct = sum(
        prediction(model, (first, last)) == target
        for first, last, target in BRANCHES
    )
    reversed_rejected = sum(
        prediction(model, (last, first)) != target
        for first, last, target in BRANCHES
    )
    left = substrate_state(model, ("A", "B"))
    right = substrate_state(model, ("D", "B"))
    context_cosine = model.assembly_cosine(left, right)

    # A third association tests whether consolidated branches survive later
    # local learning rather than only passing immediately after training.
    for iteration in range(80):
        model.learn_association(("F", "B"), "G", learning_rate=0.08)
        if iteration % 10 == 0:
            model.consolidate()
    retained_after_new = sum(
        prediction(model, (first, last)) == target
        for first, last, target in BRANCHES
    )

    return ProbeResult(
        condition=condition,
        seed=seed,
        branch_correct=branch_correct,
        reversed_rejected=reversed_rejected,
        retained_after_new=retained_after_new,
        context_cosine=context_cosine,
        edge_distance=model.mean_edge_distance(),
    )


def run_ablation(seeds: int = 5, rounds: int = 220) -> list[ProbeResult]:
    results = []
    for condition in ("random", "distance", "developed"):
        for seed in range(seeds):
            result = evaluate(condition, seed, rounds=rounds)
            results.append(result)
            print(
                f"{condition:9s} seed={seed} success={int(result.success)} "
                f"branch={result.branch_correct}/2 reverse={result.reversed_rejected}/2 "
                f"retain={result.retained_after_new}/2 "
                f"Rcos={result.context_cosine:.3f} edge_d={result.edge_distance:.3f}",
                flush=True,
            )
    print("\nsummary")
    for condition in ("random", "distance", "developed"):
        group = [result for result in results if result.condition == condition]
        print(
            f"{condition:9s} success={sum(r.success for r in group)}/{len(group)} "
            f"mean_Rcos={mean(r.context_cosine for r in group):.3f} "
            f"mean_edge_d={mean(r.edge_distance for r in group):.3f}"
        )
    return results


def verify_goal(results: list[ProbeResult], minimum_rate: float = 0.8) -> None:
    """Require reproducibility in every matched topology condition."""
    for condition in ("random", "distance", "developed"):
        group = [result for result in results if result.condition == condition]
        required = math.ceil(minimum_rate * len(group))
        successes = sum(result.success for result in group)
        if successes < required:
            raise AssertionError(
                f"{condition}: {successes}/{len(group)} successes; required {required}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    probe_results = run_ablation(seeds=args.seeds, rounds=args.rounds)
    if args.verify:
        verify_goal(probe_results)
