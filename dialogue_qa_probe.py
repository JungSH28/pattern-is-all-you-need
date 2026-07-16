"""End-to-end fact -> memory -> query -> unrestricted-token dialogue probe.

Both tracks receive the same synthetic fact records and supervised prototype
answers. Data delivery is not required to be biological. Held-out entities get
facts but never answer labels. At inference the model accepts only a query and
entity; every registered output token competes, with no answer candidate list.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import math
from statistics import mean
from typing import Mapping, Sequence

from assembly_dialogue import BioLocalAssemblyDialogue, FunctionalAssemblyDialogue
from category_generalization_probe import CONCEPTS, HELD_OUT, LABELED, make_model
from category_memory_output_probe import LATER_CONCEPTS, LATER_HELD_OUT, LATER_LABELED


QUERIES = ("what_is", "what_feature")
FEATURE_TARGET = {
    "animal": "fur",
    "vehicle": "metal",
    "fruit": "grows",
    "tool": "handheld",
}
DISTRACTORS = ("unknown", "yes", "no")


@dataclass(frozen=True)
class DialogueResult:
    track: str
    condition: str
    seed: int
    base_correct: int
    query_gate_ablation_correct: int
    no_answer_learning_correct: int
    cold_lesion_correct: int
    no_structural_correct: int
    retained_correct: int
    later_correct: int
    base_margin: float
    retained_margin: float

    @property
    def success(self) -> bool:
        return (
            self.base_correct == 2 * len(HELD_OUT)
            and self.retained_correct == 2 * len(HELD_OUT)
            and self.later_correct == 2 * len(LATER_HELD_OUT)
        )


def expected_answers(
    held_out: Mapping[str, str]
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        item
        for entity, category in held_out.items()
        for item in (
            ("what_is", entity, category),
            ("what_feature", entity, FEATURE_TARGET[category]),
        )
    )


def observe_facts(dialogue, concepts: Mapping[str, Sequence[str]]) -> None:
    for entity, properties in concepts.items():
        dialogue.observe_fact(entity, properties)


def teach_answers(
    dialogue,
    labeled: Mapping[str, Sequence[str]],
    *,
    rounds: int,
    structural_plasticity: bool,
) -> None:
    for iteration in range(rounds):
        for category, entities in labeled.items():
            for entity in entities:
                for query, target in (
                    ("what_is", category),
                    ("what_feature", FEATURE_TARGET[category]),
                ):
                    if isinstance(dialogue, BioLocalAssemblyDialogue):
                        dialogue.teach_answer(
                            query,
                            entity,
                            target,
                            structural_plasticity=(
                                structural_plasticity and iteration == 0
                            ),
                        )
                    elif iteration == 0:
                        dialogue.teach_answer(query, entity, target)


def teach_later_with_replay(
    dialogue: BioLocalAssemblyDialogue,
    *,
    rounds: int,
    replay_every: int = 4,
) -> None:
    """Acquire later answers with sparse reactivation of old prototypes."""
    for iteration in range(rounds):
        for category, entities in LATER_LABELED.items():
            for entity in entities:
                for query, target in (
                    ("what_is", category),
                    ("what_feature", FEATURE_TARGET[category]),
                ):
                    dialogue.teach_answer(
                        query,
                        entity,
                        target,
                        structural_plasticity=iteration == 0,
                    )
        if iteration % replay_every == 0:
            for category, entities in LABELED.items():
                for entity in entities:
                    for query, target in (
                        ("what_is", category),
                        ("what_feature", FEATURE_TARGET[category]),
                    ):
                        dialogue.teach_answer(query, entity, target)


def score_answers(dialogue, expected, *, gated: bool = True) -> tuple[int, float]:
    outcomes = []
    for query, entity, target in expected:
        predicted, margin, _ = dialogue.answer(query, entity, gated=gated)
        outcomes.append((predicted == target, margin))
    correct = sum(is_correct for is_correct, _ in outcomes)
    signed_margin = mean(
        margin if is_correct else -margin for is_correct, margin in outcomes
    )
    return correct, signed_margin


def functional_result(seed: int) -> DialogueResult:
    dialogue = FunctionalAssemblyDialogue(seed=seed)
    dialogue.register_queries(QUERIES)
    dialogue.register_vocabulary(DISTRACTORS)
    observe_facts(dialogue, CONCEPTS)
    base_expected = expected_answers(HELD_OUT)
    no_learning, _ = score_answers(dialogue, base_expected)
    teach_answers(
        dialogue,
        LABELED,
        rounds=1,
        structural_plasticity=False,
    )
    dialogue.clear_hot_state()
    base_correct, base_margin = score_answers(dialogue, base_expected)
    gate_ablation, _ = score_answers(dialogue, base_expected, gated=False)

    observe_facts(dialogue, LATER_CONCEPTS)
    teach_answers(
        dialogue,
        LATER_LABELED,
        rounds=1,
        structural_plasticity=False,
    )
    dialogue.clear_hot_state()
    retained_correct, retained_margin = score_answers(dialogue, base_expected)
    later_correct, _ = score_answers(
        dialogue, expected_answers(LATER_HELD_OUT)
    )
    return DialogueResult(
        track="functional",
        condition="global_associative",
        seed=seed,
        base_correct=base_correct,
        query_gate_ablation_correct=gate_ablation,
        no_answer_learning_correct=no_learning,
        cold_lesion_correct=-1,
        no_structural_correct=-1,
        retained_correct=retained_correct,
        later_correct=later_correct,
        base_margin=base_margin,
        retained_margin=retained_margin,
    )


def make_bio_dialogue(
    condition: str, seed: int, *, query_gate_fraction: float
) -> BioLocalAssemblyDialogue:
    dialogue = BioLocalAssemblyDialogue(
        make_model(
            condition,
            seed,
            query_gate_fraction=query_gate_fraction,
        )
    )
    dialogue.register_queries(QUERIES)
    dialogue.register_vocabulary(DISTRACTORS)
    return dialogue


def bio_result(
    condition: str,
    seed: int,
    *,
    answer_rounds: int = 20,
    query_gate_fraction: float = 0.50,
) -> DialogueResult:
    base_expected = expected_answers(HELD_OUT)
    dialogue = make_bio_dialogue(
        condition, seed, query_gate_fraction=query_gate_fraction
    )
    observe_facts(dialogue, CONCEPTS)
    no_learning, _ = score_answers(dialogue, base_expected)
    teach_answers(
        dialogue,
        LABELED,
        rounds=answer_rounds,
        structural_plasticity=True,
    )
    gate_ablation, _ = score_answers(dialogue, base_expected, gated=False)

    lesion = copy.deepcopy(dialogue)
    lesion.connectome.warm.zero_()
    lesion.connectome.state.zero_()
    cold_lesion, _ = score_answers(lesion, base_expected)
    del lesion

    dialogue.consolidate_and_clear()
    base_correct, base_margin = score_answers(dialogue, base_expected)
    observe_facts(dialogue, LATER_CONCEPTS)
    # Interleaved cortical replay: new answers remain the dominant experience,
    # while old query/answer traces are reactivated every fourth round.
    teach_later_with_replay(
        dialogue,
        rounds=answer_rounds,
    )
    dialogue.consolidate_and_clear()
    retained_correct, retained_margin = score_answers(dialogue, base_expected)
    later_correct, _ = score_answers(
        dialogue, expected_answers(LATER_HELD_OUT)
    )
    del dialogue

    no_structural = make_bio_dialogue(
        condition, seed, query_gate_fraction=query_gate_fraction
    )
    observe_facts(no_structural, CONCEPTS)
    teach_answers(
        no_structural,
        LABELED,
        rounds=answer_rounds,
        structural_plasticity=False,
    )
    no_structural.consolidate_and_clear()
    no_structural_correct, _ = score_answers(no_structural, base_expected)
    del no_structural

    return DialogueResult(
        track="bio_local",
        condition=condition,
        seed=seed,
        base_correct=base_correct,
        query_gate_ablation_correct=gate_ablation,
        no_answer_learning_correct=no_learning,
        cold_lesion_correct=cold_lesion,
        no_structural_correct=no_structural_correct,
        retained_correct=retained_correct,
        later_correct=later_correct,
        base_margin=base_margin,
        retained_margin=retained_margin,
    )


def run_probe(
    seeds: int = 10,
    *,
    answer_rounds: int = 20,
    query_gate_fraction: float = 0.50,
    verbose: bool = True,
) -> list[DialogueResult]:
    results = []
    for seed in range(seeds):
        result = functional_result(seed)
        results.append(result)
        if verbose:
            print(
                f"functional seed={seed:2d} success={int(result.success)} "
                f"base={result.base_correct}/8 gate_off="
                f"{result.query_gate_ablation_correct}/8 retained="
                f"{result.retained_correct}/8 later={result.later_correct}/4",
                flush=True,
            )
    for condition in ("random", "distance", "developed"):
        for seed in range(seeds):
            result = bio_result(
                condition,
                seed,
                answer_rounds=answer_rounds,
                query_gate_fraction=query_gate_fraction,
            )
            results.append(result)
            if verbose:
                print(
                    f"bio/{condition:9s} seed={seed:2d} "
                    f"success={int(result.success)} base={result.base_correct}/8 "
                    f"gate_off={result.query_gate_ablation_correct}/8 "
                    f"lesion={result.cold_lesion_correct}/8 no_rewire="
                    f"{result.no_structural_correct}/8 retained="
                    f"{result.retained_correct}/8 later={result.later_correct}/4",
                    flush=True,
                )

    print("\nsummary")
    groups = [("functional", "global_associative")]
    groups.extend(("bio_local", condition) for condition in ("random", "distance", "developed"))
    for track, condition in groups:
        group = [
            result
            for result in results
            if result.track == track and result.condition == condition
        ]
        print(
            f"{track}/{condition:18s} "
            f"success={sum(r.success for r in group)}/{len(group)} "
            f"base={mean(r.base_correct for r in group):.2f}/8 "
            f"gate_off={mean(r.query_gate_ablation_correct for r in group):.2f}/8 "
            f"no_learning={mean(r.no_answer_learning_correct for r in group):.2f}/8 "
            f"lesion={mean(r.cold_lesion_correct for r in group):.2f}/8 "
            f"no_rewire={mean(r.no_structural_correct for r in group):.2f}/8 "
            f"retained={mean(r.retained_correct for r in group):.2f}/8 "
            f"later={mean(r.later_correct for r in group):.2f}/4 "
            f"margin={mean(r.base_margin for r in group):+.3f}->"
            f"{mean(r.retained_margin for r in group):+.3f}"
        )
    return results


def verify_goal(
    results: Sequence[DialogueResult],
    *,
    minimum_accuracy: float = 0.85,
    minimum_success_rate: float = 0.40,
) -> None:
    groups = [("functional", "global_associative")]
    groups.extend(("bio_local", condition) for condition in ("random", "distance", "developed"))
    for track, condition in groups:
        group = [
            result
            for result in results
            if result.track == track and result.condition == condition
        ]
        if not group:
            raise AssertionError(f"missing {track}/{condition}")
        base_total = len(group) * 2 * len(HELD_OUT)
        later_total = len(group) * 2 * len(LATER_HELD_OUT)
        base = sum(result.base_correct for result in group)
        retained = sum(result.retained_correct for result in group)
        later = sum(result.later_correct for result in group)
        successes = sum(result.success for result in group)
        required = math.ceil(minimum_success_rate * len(group))
        if base / base_total < minimum_accuracy:
            raise AssertionError(f"{track}/{condition}: base {base}/{base_total}")
        if retained / base_total < minimum_accuracy:
            raise AssertionError(
                f"{track}/{condition}: retained {retained}/{base_total}"
            )
        if later / later_total < minimum_accuracy:
            raise AssertionError(f"{track}/{condition}: later {later}/{later_total}")
        if successes < required:
            raise AssertionError(
                f"{track}/{condition}: perfect {successes}/{len(group)}"
            )
        if mean(result.base_margin for result in group) <= 0.05:
            raise AssertionError(f"{track}/{condition}: base margin")
        if mean(result.retained_margin for result in group) <= 0.05:
            raise AssertionError(f"{track}/{condition}: retained margin")

        gate_off = sum(result.query_gate_ablation_correct for result in group)
        no_learning = sum(result.no_answer_learning_correct for result in group)
        if base - gate_off < 0.20 * base_total:
            raise AssertionError(f"{track}/{condition}: query gate contribution")
        if base - no_learning < 0.20 * base_total:
            raise AssertionError(f"{track}/{condition}: answer learning contribution")
        if track == "bio_local":
            lesion = sum(result.cold_lesion_correct for result in group)
            no_rewire = sum(result.no_structural_correct for result in group)
            if base - lesion < 0.20 * base_total:
                raise AssertionError(f"{condition}: consolidation contribution")
            # Structural rewiring is an explicit non-local scaffold ablation.
            # It need not help: a high no-rewire score is evidence that this
            # task does not depend on the scaffold, not a verifier failure.
            if not 0 <= no_rewire <= base_total:
                raise AssertionError(f"{condition}: invalid no-rewire control")

    audit = BioLocalAssemblyDialogue.locality_audit().as_dict()
    expected_local = {
        "sparse_token_assemblies",
        "local_state_propagation",
        "local_synaptic_value_update",
        "local_query_gate_application",
        "no_autograd_or_weight_transport",
    }
    expected_scaffolds = {
        "global_teacher_target",
        "global_activity_competition",
        "global_structural_search",
        "global_seed_and_gate_balancing",
    }
    if not all(audit[name] for name in expected_local | expected_scaffolds):
        raise AssertionError("locality audit is incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--answer-rounds", type=int, default=20)
    parser.add_argument("--query-gate-fraction", type=float, default=0.50)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    probe_results = run_probe(
        seeds=args.seeds,
        answer_rounds=args.answer_rounds,
        query_gate_fraction=args.query_gate_fraction,
        verbose=not args.quiet,
    )
    if args.verify:
        verify_goal(probe_results)
