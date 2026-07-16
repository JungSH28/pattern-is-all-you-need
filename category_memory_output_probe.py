"""Connect learned category geometry to cold memory and language output.

Only labeled prototypes ever co-occur with the category output tokens. Held-out
entities receive property experience but no animal/vehicle teacher. A success
therefore requires the learned R geometry to transfer through local R->O
synapses and activate the correct O token assembly.

The probe separates acquisition from long-term recall:

1. learn category outputs in warm synapses;
2. lesion warm memory before consolidation as a control;
3. consolidate, erase warm activity and hot state, then recall from cold alone;
4. learn two later categories and test both new learning and old retention.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import math
from statistics import mean
from typing import Mapping, Sequence

from category_generalization_probe import (
    CONCEPTS,
    HELD_OUT,
    LABELED,
    SWAPPED_PROPERTIES,
    make_model,
)
from spatial_connectome import OUTPUT, SUBSTRATE, SpatialConnectome


LATER_CONCEPTS = {
    "apple": ("sweet", "grows", "seeds"),
    "pear": ("sweet", "grows", "green"),
    "plum": ("sweet", "grows", "purple"),
    "peach": ("sweet", "grows", "soft"),
    "hammer": ("metal", "handheld", "hits"),
    "saw": ("metal", "handheld", "cuts"),
    "wrench": ("metal", "handheld", "turns"),
    "drill": ("metal", "handheld", "drills"),
}
LATER_LABELED = {
    "fruit": ("apple", "pear", "plum"),
    "tool": ("hammer", "saw", "wrench"),
}
LATER_HELD_OUT = {"peach": "fruit", "drill": "tool"}


@dataclass(frozen=True)
class MemoryOutputResult:
    condition: str
    seed: int
    no_output_correct: int
    warm_correct: int
    unconsolidated_lesion_correct: int
    cold_correct: int
    labeled_only_correct: int
    swapped_feature_correct: int
    retained_correct: int
    later_correct: int
    cold_margin: float
    retained_margin: float
    cold_output_change: float

    @property
    def success(self) -> bool:
        return (
            self.cold_correct == len(HELD_OUT)
            and self.retained_correct == len(HELD_OUT)
            and self.later_correct == len(LATER_HELD_OUT)
        )


def learn_concepts(
    model: SpatialConnectome, concepts: Mapping[str, Sequence[str]]
) -> None:
    for entity, properties in concepts.items():
        model.learn_concept(entity, properties)


def teach_outputs(
    model: SpatialConnectome,
    labeled: Mapping[str, Sequence[str]],
    *,
    rounds: int = 20,
) -> None:
    categories = tuple(labeled)
    model.register_vocabulary(categories)
    for category, entities in labeled.items():
        for entity in entities:
            model.learn_output_association(
                (entity,), category, structural_plasticity=True
            )
    for _ in range(rounds - 1):
        for category, entities in labeled.items():
            for entity in entities:
                model.learn_output_association((entity,), category)


def predictions(
    model: SpatialConnectome,
    expected: Mapping[str, str],
    candidates: Sequence[str],
) -> tuple[int, float]:
    outcomes = []
    for entity, target in expected.items():
        predicted, margin, _ = model.predict_output((entity,), candidates)
        outcomes.append((predicted == target, margin))
    correct = sum(is_correct for is_correct, _ in outcomes)
    signed_margin = mean(
        margin if is_correct else -margin for is_correct, margin in outcomes
    )
    return correct, signed_margin


def consolidate_and_erase_fast_state(model: SpatialConnectome) -> None:
    model.consolidate(cycles=100)
    model.warm.zero_()
    model.state.zero_()


def evaluate(
    condition: str,
    seed: int,
    *,
    out_degree: int = 192,
    output_rounds: int = 20,
) -> MemoryOutputResult:
    categories = tuple(LABELED)

    model = make_model(condition, seed, out_degree=out_degree)
    learn_concepts(model, CONCEPTS)
    model.register_vocabulary(categories)
    no_output_correct, _ = predictions(model, HELD_OUT, categories)

    output_edge = (model.region[model.src] == SUBSTRATE) & (
        model.region[model.dst] == OUTPUT
    )
    cold_before = model.cold[output_edge].clone()
    teach_outputs(model, LABELED, rounds=output_rounds)
    warm_correct, _ = predictions(model, HELD_OUT, categories)

    lesion = copy.deepcopy(model)
    lesion.warm.zero_()
    lesion.state.zero_()
    unconsolidated_lesion_correct, _ = predictions(lesion, HELD_OUT, categories)
    del lesion

    consolidate_and_erase_fast_state(model)
    cold_correct, cold_margin = predictions(model, HELD_OUT, categories)
    cold_output_change = float(
        (model.cold[output_edge] - cold_before).abs().mean().item()
    )

    # A later learning session adds two categories. Old recall competes against
    # all four output words, so retention cannot pass through a closed decoder.
    learn_concepts(model, LATER_CONCEPTS)
    teach_outputs(model, LATER_LABELED, rounds=output_rounds)
    consolidate_and_erase_fast_state(model)
    all_categories = (*categories, *tuple(LATER_LABELED))
    retained_correct, retained_margin = predictions(
        model, HELD_OUT, all_categories
    )
    later_correct, _ = predictions(model, LATER_HELD_OUT, all_categories)
    del model

    # Control: output labels are taught, but held-out R assemblies never receive
    # property experience and therefore have no learned category geometry.
    labeled_only = make_model(condition, seed, out_degree=out_degree)
    labeled_entities = {
        entity: CONCEPTS[entity]
        for entities in LABELED.values()
        for entity in entities
    }
    learn_concepts(labeled_only, labeled_entities)
    teach_outputs(labeled_only, LABELED, rounds=output_rounds)
    consolidate_and_erase_fast_state(labeled_only)
    labeled_only_correct, _ = predictions(labeled_only, HELD_OUT, categories)
    del labeled_only

    # Counterfactual: keep each held-out name but give it the opposite category's
    # properties. Language output should follow properties and flip the word.
    swapped = make_model(condition, seed, out_degree=out_degree)
    swapped_concepts = {
        entity: SWAPPED_PROPERTIES.get(entity, properties)
        for entity, properties in CONCEPTS.items()
    }
    learn_concepts(swapped, swapped_concepts)
    teach_outputs(swapped, LABELED, rounds=output_rounds)
    consolidate_and_erase_fast_state(swapped)
    swapped_predictions = [
        swapped.predict_output((entity,), categories)[0]
        for entity in HELD_OUT
    ]
    swapped_feature_correct = sum(
        predicted != HELD_OUT[entity]
        for entity, predicted in zip(HELD_OUT, swapped_predictions)
    )
    del swapped

    return MemoryOutputResult(
        condition=condition,
        seed=seed,
        no_output_correct=no_output_correct,
        warm_correct=warm_correct,
        unconsolidated_lesion_correct=unconsolidated_lesion_correct,
        cold_correct=cold_correct,
        labeled_only_correct=labeled_only_correct,
        swapped_feature_correct=swapped_feature_correct,
        retained_correct=retained_correct,
        later_correct=later_correct,
        cold_margin=cold_margin,
        retained_margin=retained_margin,
        cold_output_change=cold_output_change,
    )


def run_ablation(
    seeds: int = 10,
    *,
    out_degree: int = 192,
    output_rounds: int = 20,
    verbose: bool = True,
) -> list[MemoryOutputResult]:
    results = []
    for condition in ("random", "distance", "developed"):
        for seed in range(seeds):
            result = evaluate(
                condition,
                seed,
                out_degree=out_degree,
                output_rounds=output_rounds,
            )
            results.append(result)
            if verbose:
                print(
                    f"{condition:9s} seed={seed:2d} success={int(result.success)} "
                    f"cold={result.cold_correct}/4 lesion="
                    f"{result.unconsolidated_lesion_correct}/4 "
                    f"retained={result.retained_correct}/4 later="
                    f"{result.later_correct}/2 swapped="
                    f"{result.swapped_feature_correct}/4 "
                    f"margin={result.cold_margin:+.3f}->"
                    f"{result.retained_margin:+.3f}",
                    flush=True,
                )
    print("\nsummary")
    for condition in ("random", "distance", "developed"):
        group = [result for result in results if result.condition == condition]
        print(
            f"{condition:9s} success={sum(r.success for r in group)}/{len(group)} "
            f"warm={mean(r.warm_correct for r in group):.2f}/4 "
            f"cold={mean(r.cold_correct for r in group):.2f}/4 "
            f"no_output={mean(r.no_output_correct for r in group):.2f}/4 "
            f"lesion={mean(r.unconsolidated_lesion_correct for r in group):.2f}/4 "
            f"labeled_only={mean(r.labeled_only_correct for r in group):.2f}/4 "
            f"swapped={mean(r.swapped_feature_correct for r in group):.2f}/4 "
            f"retained={mean(r.retained_correct for r in group):.2f}/4 "
            f"later={mean(r.later_correct for r in group):.2f}/2 "
            f"margin={mean(r.cold_margin for r in group):+.3f}->"
            f"{mean(r.retained_margin for r in group):+.3f}"
        )
    return results


def verify_goal(
    results: Sequence[MemoryOutputResult],
    *,
    minimum_accuracy: float = 0.85,
    minimum_success_rate: float = 0.50,
) -> None:
    for condition in ("random", "distance", "developed"):
        group = [result for result in results if result.condition == condition]
        if not group:
            raise AssertionError(f"missing condition: {condition}")
        old_total = len(group) * len(HELD_OUT)
        later_total = len(group) * len(LATER_HELD_OUT)
        required = math.ceil(minimum_success_rate * len(group))
        successes = sum(result.success for result in group)
        warm = sum(result.warm_correct for result in group)
        cold = sum(result.cold_correct for result in group)
        retained = sum(result.retained_correct for result in group)
        later = sum(result.later_correct for result in group)
        controls = max(
            sum(result.no_output_correct for result in group),
            sum(result.unconsolidated_lesion_correct for result in group),
            sum(result.labeled_only_correct for result in group),
        )
        swapped = sum(result.swapped_feature_correct for result in group)
        if successes < required:
            raise AssertionError(
                f"{condition}: {successes}/{len(group)} complete seeds; "
                f"required {required}"
            )
        if warm / old_total < minimum_accuracy:
            raise AssertionError(f"{condition}: warm acquisition {warm}/{old_total}")
        if cold / old_total < minimum_accuracy:
            raise AssertionError(f"{condition}: cold recall {cold}/{old_total}")
        if retained / old_total < minimum_accuracy:
            raise AssertionError(
                f"{condition}: post-interference retention {retained}/{old_total}"
            )
        if later / later_total < minimum_accuracy:
            raise AssertionError(f"{condition}: later learning {later}/{later_total}")
        if swapped / old_total < minimum_accuracy:
            raise AssertionError(
                f"{condition}: counterfactual output {swapped}/{old_total}"
            )
        if cold - controls < 0.20 * old_total:
            raise AssertionError(
                f"{condition}: cold-control gain {cold - controls}/{old_total} < 20%"
            )
        if mean(result.cold_margin for result in group) <= 0.05:
            raise AssertionError(f"{condition}: cold output margin did not emerge")
        if mean(result.retained_margin for result in group) <= 0.05:
            raise AssertionError(f"{condition}: retained output margin did not survive")
        if min(result.cold_output_change for result in group) <= 0.0:
            raise AssertionError(f"{condition}: no R->O change reached cold memory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out-degree", type=int, default=192)
    parser.add_argument("--output-rounds", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    probe_results = run_ablation(
        seeds=args.seeds,
        out_degree=args.out_degree,
        output_rounds=args.output_rounds,
        verbose=not args.quiet,
    )
    if args.verify:
        verify_goal(probe_results)
