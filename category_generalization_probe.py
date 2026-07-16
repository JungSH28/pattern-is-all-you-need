"""Held-out category generalization in the single sparse connectome.

Category membership is available only for cat/dog/horse and car/bus/van.
Wolf, fox, truck, and bike are exposed to properties but never to a category
target. The
probe asks whether their R assemblies are nearer (cosine) to the correct
category prototype than the other prototype.

This separates two geometries from the project thesis:

* 3-D Euclidean distance affects which connectome edges exist;
* R-assembly cosine measures conceptual similarity and category transfer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from statistics import mean

import torch

from spatial_connectome import SUBSTRATE, ConnectomeConfig, SpatialConnectome


CONCEPTS = {
    "cat": ("fur", "fourlegs", "meows"),
    "dog": ("fur", "fourlegs", "barks"),
    "horse": ("fur", "fourlegs", "neighs"),
    "wolf": ("fur", "fourlegs", "howls"),
    "fox": ("fur", "fourlegs", "wild"),
    "car": ("metal", "wheels", "engine"),
    "bus": ("metal", "wheels", "seats"),
    "van": ("metal", "wheels", "doors"),
    "truck": ("metal", "wheels", "cargo"),
    # Every held-out item shares two category properties and has one novel
    # property. This balances the animal and vehicle transfer difficulty.
    "bike": ("metal", "wheels", "pedals"),
}
LABELED = {
    "animal": ("cat", "dog", "horse"),
    "vehicle": ("car", "bus", "van"),
}
HELD_OUT = {
    "wolf": "animal",
    "fox": "animal",
    "truck": "vehicle",
    "bike": "vehicle",
}
SWAPPED_PROPERTIES = {
    "wolf": CONCEPTS["truck"],
    "fox": CONCEPTS["bike"],
    "truck": CONCEPTS["wolf"],
    "bike": CONCEPTS["fox"],
}


@dataclass(frozen=True)
class CategoryResult:
    condition: str
    seed: int
    learned_correct: int
    baseline_correct: int
    labeled_only_correct: int
    swapped_feature_correct: int
    learned_margin: float
    geometry_gap: float
    edge_distance: float

    @property
    def success(self) -> bool:
        return self.learned_correct == len(HELD_OUT) and self.learned_margin > 0.0


def vocabulary() -> tuple[str, ...]:
    tokens = set(CONCEPTS)
    for properties in CONCEPTS.values():
        tokens.update(properties)
    return tuple(sorted(tokens))


def make_model(
    condition: str,
    seed: int,
    *,
    out_degree: int = 192,
    query_gate_fraction: float = 0.50,
    structural_rewire_mode: str = "weakest",
) -> SpatialConnectome:
    topology = "random" if condition == "random" else "distance"
    model = SpatialConnectome(
        ConnectomeConfig(
            # Keep sensory seed collisions low enough that structural plasticity
            # for one name does not rewrite another name's I->R routes.
            n_input=256,
            n_substrate=512,
            n_output=64,
            out_degree=out_degree,
            topology=topology,
            steps_per_token=2,
            max_region_density=0.05,
            max_output_density=0.10,
            initial_weight=0.08,
            query_gate_fraction=query_gate_fraction,
            structural_rewire_mode=structural_rewire_mode,
            seed=seed,
        )
    )
    model.register_vocabulary(vocabulary())
    if condition == "developed":
        traces = []
        for entity, properties in CONCEPTS.items():
            for _ in range(2):
                state = model.simultaneous_state((entity, *properties), steps=2)
                traces.append(state)
        model.develop_positions(torch.stack(traces), epochs=3)
    return model


def concept_state(model: SpatialConnectome, entity: str) -> torch.Tensor:
    state = model.simultaneous_state((entity,), steps=2)
    return state[model.region == SUBSTRATE]


def prototypes(model: SpatialConnectome) -> dict[str, torch.Tensor]:
    return {
        category: torch.stack([concept_state(model, item) for item in items]).mean(0)
        for category, items in LABELED.items()
    }


def classify(
    model: SpatialConnectome, entity: str, category_prototypes: dict[str, torch.Tensor]
) -> tuple[str, float]:
    state = concept_state(model, entity)
    scores = {
        category: model.assembly_cosine(state, prototype)
        for category, prototype in category_prototypes.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ordered[0][0], ordered[0][1] - ordered[1][1]


def representation_gap(model: SpatialConnectome) -> float:
    """Mean within-category cosine minus mean cross-category cosine."""
    animal = [concept_state(model, item) for item in CONCEPTS if item in ("cat", "dog", "wolf", "fox")]
    vehicle = [concept_state(model, item) for item in CONCEPTS if item in ("car", "bus", "truck", "bike")]
    within = []
    for group in (animal, vehicle):
        for i in range(len(group)):
            for j in range(i):
                within.append(model.assembly_cosine(group[i], group[j]))
    cross = [model.assembly_cosine(a, v) for a in animal for v in vehicle]
    return mean(within) - mean(cross)


def evaluate(
    condition: str,
    seed: int,
    *,
    out_degree: int = 192,
    concept_rounds: int = 40,
) -> CategoryResult:
    baseline = make_model(condition, seed, out_degree=out_degree)
    baseline_prototypes = prototypes(baseline)
    baseline_correct = sum(
        classify(baseline, entity, baseline_prototypes)[0] == category
        for entity, category in HELD_OUT.items()
    )
    del baseline, baseline_prototypes

    # Stronger control: organize the labeled prototypes from properties, but
    # leave all held-out entity assemblies untouched/random.
    labeled_only = make_model(condition, seed, out_degree=out_degree)
    labeled_items = tuple(
        item for items in LABELED.values() for item in items
    )
    for entity in labeled_items:
        labeled_only.learn_concept(
            entity, CONCEPTS[entity], rounds=concept_rounds
        )
    labeled_only_prototypes = prototypes(labeled_only)
    labeled_only_correct = sum(
        classify(labeled_only, entity, labeled_only_prototypes)[0] == category
        for entity, category in HELD_OUT.items()
    )
    del labeled_only, labeled_only_prototypes

    # Every entity, including held-out items, receives property experience.
    # Only the six LABELED items define category prototypes afterward.
    learned = make_model(condition, seed, out_degree=out_degree)
    for entity, properties in CONCEPTS.items():
        learned.learn_concept(entity, properties, rounds=concept_rounds)
    learned_prototypes = prototypes(learned)
    predictions = [
        (entity, category, *classify(learned, entity, learned_prototypes))
        for entity, category in HELD_OUT.items()
    ]
    learned_correct = sum(predicted == category for _, category, predicted, _ in predictions)
    learned_margin = mean(
        margin if predicted == category else -margin
        for _, category, predicted, margin in predictions
    )
    geometry_gap = representation_gap(learned)
    edge_distance = learned.mean_edge_distance()
    del learned, learned_prototypes, predictions

    # Counterfactual control: retain the same held-out names but swap their
    # property experiences across categories. Predictions should follow the
    # properties, hence the opposite of each original HELD_OUT label.
    swapped = make_model(condition, seed, out_degree=out_degree)
    for entity, properties in CONCEPTS.items():
        swapped.learn_concept(
            entity,
            SWAPPED_PROPERTIES.get(entity, properties),
            rounds=concept_rounds,
        )
    swapped_prototypes = prototypes(swapped)
    swapped_feature_correct = sum(
        classify(swapped, entity, swapped_prototypes)[0] != original_category
        for entity, original_category in HELD_OUT.items()
    )
    del swapped, swapped_prototypes

    return CategoryResult(
        condition=condition,
        seed=seed,
        learned_correct=learned_correct,
        baseline_correct=baseline_correct,
        labeled_only_correct=labeled_only_correct,
        swapped_feature_correct=swapped_feature_correct,
        learned_margin=learned_margin,
        geometry_gap=geometry_gap,
        edge_distance=edge_distance,
    )


def run_ablation(
    seeds: int = 10, *, out_degree: int = 192, verbose: bool = True
) -> list[CategoryResult]:
    results = []
    for condition in ("random", "distance", "developed"):
        for seed in range(seeds):
            result = evaluate(condition, seed, out_degree=out_degree)
            results.append(result)
            if verbose:
                print(
                    f"{condition:9s} seed={seed:2d} success={int(result.success)} "
                    f"heldout={result.learned_correct}/4 baseline={result.baseline_correct}/4 "
                    f"labeled_only={result.labeled_only_correct}/4 "
                    f"swapped={result.swapped_feature_correct}/4 "
                    f"margin={result.learned_margin:+.3f} gap={result.geometry_gap:+.3f} "
                    f"edge_d={result.edge_distance:.3f}",
                    flush=True,
                )
    print("\nsummary")
    for condition in ("random", "distance", "developed"):
        group = [result for result in results if result.condition == condition]
        print(
            f"{condition:9s} success={sum(r.success for r in group)}/{len(group)} "
            f"heldout={mean(r.learned_correct for r in group):.2f}/4 "
            f"baseline={mean(r.baseline_correct for r in group):.2f}/4 "
            f"labeled_only={mean(r.labeled_only_correct for r in group):.2f}/4 "
            f"swapped={mean(r.swapped_feature_correct for r in group):.2f}/4 "
            f"margin={mean(r.learned_margin for r in group):+.3f} "
            f"gap={mean(r.geometry_gap for r in group):+.3f}"
        )
    return results


def verify_goal(
    results: list[CategoryResult],
    minimum_accuracy: float = 0.85,
    minimum_perfect_seed_rate: float = 0.60,
) -> None:
    for condition in ("random", "distance", "developed"):
        group = [result for result in results if result.condition == condition]
        required = math.ceil(minimum_perfect_seed_rate * len(group))
        successes = sum(result.success for result in group)
        learned_accuracy = sum(result.learned_correct for result in group)
        baseline_accuracy = sum(result.baseline_correct for result in group)
        labeled_only_accuracy = sum(result.labeled_only_correct for result in group)
        swapped_accuracy = sum(result.swapped_feature_correct for result in group)
        total = len(group) * len(HELD_OUT)
        if successes < required:
            raise AssertionError(
                f"{condition}: {successes}/{len(group)} seeds; required {required}"
            )
        if learned_accuracy / total < minimum_accuracy:
            raise AssertionError(
                f"{condition}: held-out accuracy {learned_accuracy}/{total} "
                f"< {minimum_accuracy:.0%}"
            )
        if swapped_accuracy / total < minimum_accuracy:
            raise AssertionError(
                f"{condition}: swapped-feature accuracy {swapped_accuracy}/{total} "
                f"< {minimum_accuracy:.0%}"
            )
        if learned_accuracy <= baseline_accuracy:
            raise AssertionError(
                f"{condition}: learned {learned_accuracy} <= baseline {baseline_accuracy}"
            )
        if learned_accuracy <= labeled_only_accuracy:
            raise AssertionError(
                f"{condition}: learned {learned_accuracy} <= labeled-only "
                f"{labeled_only_accuracy}"
            )
        strongest_control = max(baseline_accuracy, labeled_only_accuracy)
        if learned_accuracy - strongest_control < 0.20 * total:
            raise AssertionError(
                f"{condition}: learned-control gain "
                f"{learned_accuracy - strongest_control}/{total} < 20%"
            )
        if mean(result.geometry_gap for result in group) <= 0.05:
            raise AssertionError(f"{condition}: category geometry gap did not emerge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out-degree", type=int, default=192)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    probe_results = run_ablation(
        seeds=args.seeds, out_degree=args.out_degree, verbose=not args.quiet
    )
    if args.verify:
        verify_goal(probe_results)
