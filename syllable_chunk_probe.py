"""Boundary-free Korean syllable stream to unrestricted O-token probe.

The model sees concatenated Hangul syllables and supervised fact/answer
targets.  Hidden word boundaries and semantic query types exist only in this
evaluator.  They are never passed through either model API.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import math
from statistics import mean
from typing import Mapping, Sequence

from category_generalization_probe import CONCEPTS, HELD_OUT, LABELED
from category_memory_output_probe import (
    LATER_CONCEPTS,
    LATER_HELD_OUT,
    LATER_LABELED,
)
from syllable_chunk_dialogue import (
    BioLocalSyllableDialogue,
    ConnectomeSyllableDialogue,
    FunctionalSyllableDialogue,
    syllable_sequence,
)


SURFACE = {
    "cat": "고양이",
    "dog": "강아지",
    "horse": "조랑말",
    "wolf": "늑대",
    "fox": "여우",
    "car": "자동차",
    "bus": "버스",
    "van": "승합차",
    "truck": "트럭",
    "bike": "자전거",
    "apple": "사과",
    "pear": "서양배",
    "plum": "자두",
    "peach": "복숭아",
    "hammer": "망치",
    "saw": "전기톱",
    "wrench": "렌치",
    "drill": "드릴",
}

PROPERTY_SURFACE = {
    "fur": "털",
    "fourlegs": "네발",
    "meows": "야옹소리",
    "barks": "멍멍소리",
    "neighs": "말울음",
    "howls": "하울링",
    "wild": "야생성",
    "metal": "금속",
    "wheels": "바퀴",
    "engine": "엔진",
    "seats": "좌석",
    "doors": "문짝",
    "cargo": "화물",
    "pedals": "페달",
    "sweet": "단맛",
    "grows": "재배",
    "seeds": "씨앗",
    "green": "초록빛",
    "purple": "보랏빛",
    "soft": "부드러움",
    "handheld": "손도구",
    "hits": "두드림",
    "cuts": "자르기",
    "turns": "돌리기",
    "drills": "구멍뚫기",
}

CATEGORY_OUTPUT = {
    "animal": "동물",
    "vehicle": "탈것",
    "fruit": "과일",
    "tool": "도구",
}

FEATURE_PROPERTY = {
    "animal": "fur",
    "vehicle": "metal",
    "fruit": "grows",
    "tool": "handheld",
}

FEATURE_OUTPUT = {
    "fur": "털",
    "metal": "금속",
    "grows": "재배",
    "handheld": "손도구",
}

DISTRACTORS = ("모름", "예", "아니오")

CATEGORY_TRAIN = (
    "{entity}는어떤종류인가",
    "무슨종류인가{entity}는",
    "{entity}의종류는무엇인가",
)
FEATURE_TRAIN = (
    "{entity}의특징은무엇인가",
    "{entity}는어떤특징이있나",
    "무슨특징인가{entity}는",
)

# These exact strings and particle combinations never occur in training.
CATEGORY_HELDOUT = "{entity}가속한종류를말해"
FEATURE_HELDOUT = "{entity}에게있는대표특징을말해"


def fact_records(
    concepts: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    records = []
    carriers = (
        "{entity}는{property}이있다",
        "{property}을가진{entity}",
        "{entity}에게{property}이보인다",
    )
    for entity, properties in concepts.items():
        for index, property_name in enumerate(properties):
            records.append(
                (
                    carriers[index % len(carriers)].format(
                        entity=SURFACE[entity],
                        property=PROPERTY_SURFACE[property_name],
                    ),
                    (property_name,),
                )
            )
    return tuple(records)


def training_questions(
    labeled: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    examples = []
    for category, entities in labeled.items():
        feature = FEATURE_PROPERTY[category]
        for entity in entities:
            surface = SURFACE[entity]
            examples.extend(
                (template.format(entity=surface), CATEGORY_OUTPUT[category])
                for template in CATEGORY_TRAIN
            )
            examples.extend(
                (template.format(entity=surface), FEATURE_OUTPUT[feature])
                for template in FEATURE_TRAIN
            )
    return tuple(examples)


def heldout_questions(
    expected: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    examples = []
    for entity, category in expected.items():
        surface = SURFACE[entity]
        examples.append(
            (CATEGORY_HELDOUT.format(entity=surface), CATEGORY_OUTPUT[category])
        )
        feature = FEATURE_PROPERTY[category]
        examples.append(
            (FEATURE_HELDOUT.format(entity=surface), FEATURE_OUTPUT[feature])
        )
    return tuple(examples)


def output_vocabulary() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (*CATEGORY_OUTPUT.values(), *FEATURE_OUTPUT.values(), *DISTRACTORS)
        )
    )


def stage_streams(
    concepts: Mapping[str, Sequence[str]],
    labeled: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    return tuple(text for text, _ in fact_records(concepts)) + tuple(
        text for text, _ in training_questions(labeled)
    )


def fixed_syllable_vocabulary_texts() -> tuple[str, ...]:
    return (
        *stage_streams(CONCEPTS, LABELED),
        *stage_streams(LATER_CONCEPTS, LATER_LABELED),
        *(text for text, _ in heldout_questions(HELD_OUT)),
        *(text for text, _ in heldout_questions(LATER_HELD_OUT)),
    )


def teach_stage(
    model,
    concepts: Mapping[str, Sequence[str]],
    labeled: Mapping[str, Sequence[str]],
    *,
    fact_rounds: int,
    answer_rounds: int,
    temporal: bool = True,
) -> None:
    records = fact_records(concepts)
    questions = training_questions(labeled)
    for _ in range(fact_rounds):
        for text, properties in records:
            model.observe_fact(text, properties, temporal=temporal)
    for _ in range(answer_rounds):
        for text, target in questions:
            model.teach_answer(text, target, temporal=temporal)


def score(
    model,
    examples: Sequence[tuple[str, str]],
    *,
    temporal: bool = True,
    controlled: bool = True,
) -> tuple[int, float]:
    outcomes = []
    for text, target in examples:
        predicted, margin, scores = model.answer(
            text, temporal=temporal, controlled=controlled
        )
        if not set(output_vocabulary()).issubset(scores):
            raise AssertionError("answer did not score the full O-token vocabulary")
        outcomes.append((predicted == target, margin))
    correct = sum(is_correct for is_correct, _ in outcomes)
    signed_margin = mean(
        margin if is_correct else -margin for is_correct, margin in outcomes
    )
    return correct, signed_margin


def reverse_syllables(text: str) -> str:
    return "".join(reversed(syllable_sequence(text)))


def score_reversed(model, examples: Sequence[tuple[str, str]]) -> int:
    return sum(
        model.answer(reverse_syllables(text))[0] == target
        for text, target in examples
    )


def chunk_recall(model, entities: Sequence[str]) -> int:
    return sum(
        model.chunker.pattern_code(syllable_sequence(SURFACE[entity])) is not None
        for entity in entities
    )


def fact_episodes(
    concepts: Mapping[str, Sequence[str]],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]:
    records = fact_records(concepts)
    return tuple(
        tuple(record for record in records if SURFACE[entity] in record[0])
        for entity in concepts
    )


def score_connectome(
    model: ConnectomeSyllableDialogue,
    examples: Sequence[tuple[str, str]],
    *,
    controlled: bool = True,
) -> tuple[int, float]:
    outcomes = []
    for text, target in examples:
        predicted, margin, scores = model.answer(text, controlled=controlled)
        if not set(output_vocabulary()).issubset(scores):
            raise AssertionError("answer did not score the full O-token vocabulary")
        outcomes.append((predicted == target, margin))
    correct = sum(is_correct for is_correct, _ in outcomes)
    signed_margin = mean(
        margin if is_correct else -margin for is_correct, margin in outcomes
    )
    return correct, signed_margin


@dataclass(frozen=True)
class SyllableDialogueResult:
    track: str
    seed: int
    base_correct: int
    control_ablation_correct: int
    temporal_ablation_correct: int
    reversed_correct: int
    warm_lesion_correct: int
    retained_correct: int
    later_correct: int
    base_chunks: int
    later_chunks: int
    base_margin: float
    retained_margin: float

    @property
    def success(self) -> bool:
        return (
            self.base_correct == 2 * len(HELD_OUT)
            and self.retained_correct == 2 * len(HELD_OUT)
            and self.later_correct == 2 * len(LATER_HELD_OUT)
        )


def functional_result(seed: int) -> SyllableDialogueResult:
    model = FunctionalSyllableDialogue(seed=seed)
    model.register_syllable_vocabulary(fixed_syllable_vocabulary_texts())
    model.register_output_vocabulary(output_vocabulary())
    model.observe_streams(stage_streams(CONCEPTS, LABELED))
    model.finalize_chunks()
    teach_stage(model, CONCEPTS, LABELED, fact_rounds=16, answer_rounds=16)

    base_examples = heldout_questions(HELD_OUT)
    base_correct, base_margin = score(model, base_examples)
    control_ablation, _ = score(model, base_examples, controlled=False)
    temporal_ablation, _ = score(model, base_examples, temporal=False)
    reversed_correct = score_reversed(model, base_examples)
    base_chunks = chunk_recall(model, tuple(CONCEPTS))

    model.observe_streams(stage_streams(LATER_CONCEPTS, LATER_LABELED))
    model.finalize_chunks()
    teach_stage(
        model, LATER_CONCEPTS, LATER_LABELED, fact_rounds=16, answer_rounds=16
    )
    teach_stage(model, CONCEPTS, LABELED, fact_rounds=3, answer_rounds=0)
    retained_correct, retained_margin = score(model, base_examples)
    later_correct, _ = score(model, heldout_questions(LATER_HELD_OUT))
    later_chunks = chunk_recall(model, tuple(LATER_CONCEPTS))
    return SyllableDialogueResult(
        track="functional_global",
        seed=seed,
        base_correct=base_correct,
        control_ablation_correct=control_ablation,
        temporal_ablation_correct=temporal_ablation,
        reversed_correct=reversed_correct,
        warm_lesion_correct=-1,
        retained_correct=retained_correct,
        later_correct=later_correct,
        base_chunks=base_chunks,
        later_chunks=later_chunks,
        base_margin=base_margin,
        retained_margin=retained_margin,
    )


def bio_result(
    seed: int,
    *,
    concept_rounds: int = 30,
    answer_rounds: int = 30,
    replay_rounds: int = 6,
) -> SyllableDialogueResult:
    model = ConnectomeSyllableDialogue(seed=seed)
    model.register_syllable_vocabulary(fixed_syllable_vocabulary_texts())
    model.register_output_vocabulary(output_vocabulary())
    model.observe_streams(stage_streams(CONCEPTS, LABELED))
    model.finalize_chunks()
    for episode in fact_episodes(CONCEPTS):
        model.observe_fact_episode(episode, rounds=concept_rounds)
    base_questions = training_questions(LABELED)
    for iteration in range(answer_rounds):
        for text, target in base_questions:
            model.teach_answer(
                text,
                target,
                structural_plasticity=iteration == 0,
            )
    base_examples = heldout_questions(HELD_OUT)
    control_ablation, _ = score_connectome(
        model, base_examples, controlled=False
    )
    # No local transition updates means no recurrent chunks and therefore no
    # concept episode can form. The control is executed through that failure,
    # rather than assigning hidden word boundaries to rescue it.
    no_temporal = ConnectomeSyllableDialogue(seed=seed)
    no_temporal.register_syllable_vocabulary(fixed_syllable_vocabulary_texts())
    no_temporal.observe_streams(
        stage_streams(CONCEPTS, LABELED), temporal=False
    )
    no_temporal.finalize_chunks()
    temporal_ablation = 0
    try:
        no_temporal.observe_fact_episode(
            fact_episodes(CONCEPTS)[0], rounds=concept_rounds
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("no-temporal control formed a concept chunk")
    del no_temporal
    reversed_correct = 0
    for text, target in base_examples:
        try:
            reversed_correct += (
                model.answer(reverse_syllables(text))[0] == target
            )
        except RuntimeError:
            pass
    base_chunks = chunk_recall(model, tuple(CONCEPTS))

    lesion = copy.deepcopy(model)
    lesion.connectome.warm.zero_()
    lesion.connectome.state.zero_()
    warm_lesion, _ = score_connectome(lesion, base_examples)
    del lesion

    model.consolidate_and_clear()
    base_correct, base_margin = score_connectome(model, base_examples)

    model.observe_streams(stage_streams(LATER_CONCEPTS, LATER_LABELED))
    model.finalize_chunks()
    for episode in fact_episodes(LATER_CONCEPTS):
        model.observe_fact_episode(episode, rounds=concept_rounds)
    later_questions = training_questions(LATER_LABELED)
    for iteration in range(answer_rounds):
        for text, target in later_questions:
            model.teach_answer(
                text,
                target,
                structural_plasticity=iteration == 0,
            )
        replay_interval = max(1, answer_rounds // max(1, replay_rounds))
        if iteration % replay_interval == 0:
            for text, target in base_questions:
                model.teach_answer(text, target, learn_control=False)
    model.consolidate_and_clear()
    retained_correct, retained_margin = score_connectome(model, base_examples)
    later_correct, _ = score_connectome(
        model, heldout_questions(LATER_HELD_OUT)
    )
    later_chunks = chunk_recall(model, tuple(LATER_CONCEPTS))
    return SyllableDialogueResult(
        track="bio_connectome_local",
        seed=seed,
        base_correct=base_correct,
        control_ablation_correct=control_ablation,
        temporal_ablation_correct=temporal_ablation,
        reversed_correct=reversed_correct,
        warm_lesion_correct=warm_lesion,
        retained_correct=retained_correct,
        later_correct=later_correct,
        base_chunks=base_chunks,
        later_chunks=later_chunks,
        base_margin=base_margin,
        retained_margin=retained_margin,
    )


def run_probe(seeds: int = 10, *, verbose: bool = True) -> list[SyllableDialogueResult]:
    results = []
    for seed in range(seeds):
        for evaluator in (functional_result, bio_result):
            result = evaluator(seed)
            results.append(result)
            if verbose:
                print(
                    f"{result.track:17s} seed={seed:2d} "
                    f"success={int(result.success)} base={result.base_correct}/8 "
                    f"control_off={result.control_ablation_correct}/8 "
                    f"temporal_off={result.temporal_ablation_correct}/8 "
                    f"reverse={result.reversed_correct}/8 retained="
                    f"{result.retained_correct}/8 later={result.later_correct}/4 "
                    f"chunks={result.base_chunks}/10+{result.later_chunks}/8",
                    flush=True,
                )
    print("\nsummary")
    for track in ("functional_global", "bio_connectome_local"):
        group = [result for result in results if result.track == track]
        print(
            f"{track:17s} success={sum(r.success for r in group)}/{len(group)} "
            f"base={mean(r.base_correct for r in group):.2f}/8 "
            f"control_off={mean(r.control_ablation_correct for r in group):.2f}/8 "
            f"temporal_off={mean(r.temporal_ablation_correct for r in group):.2f}/8 "
            f"reverse={mean(r.reversed_correct for r in group):.2f}/8 "
            f"lesion={mean(r.warm_lesion_correct for r in group):.2f}/8 "
            f"retained={mean(r.retained_correct for r in group):.2f}/8 "
            f"later={mean(r.later_correct for r in group):.2f}/4 "
            f"chunks={mean(r.base_chunks for r in group):.1f}/10+"
            f"{mean(r.later_chunks for r in group):.1f}/8 "
            f"margin={mean(r.base_margin for r in group):+.3f}/"
            f"{mean(r.retained_margin for r in group):+.3f}"
        )
    return results


def verify_goal(
    results: Sequence[SyllableDialogueResult],
    *,
    minimum_accuracy: float = 0.85,
    minimum_success_rate: float = 0.40,
) -> None:
    for track in ("functional_global", "bio_connectome_local"):
        group = [result for result in results if result.track == track]
        if not group:
            raise AssertionError(f"missing track: {track}")
        base_total = len(group) * 2 * len(HELD_OUT)
        later_total = len(group) * 2 * len(LATER_HELD_OUT)
        base = sum(result.base_correct for result in group)
        retained = sum(result.retained_correct for result in group)
        later = sum(result.later_correct for result in group)
        required_success = math.ceil(minimum_success_rate * len(group))
        if base / base_total < minimum_accuracy:
            raise AssertionError(f"{track}: base {base}/{base_total}")
        if retained / base_total < minimum_accuracy:
            raise AssertionError(f"{track}: retained {retained}/{base_total}")
        if later / later_total < minimum_accuracy:
            raise AssertionError(f"{track}: later {later}/{later_total}")
        if sum(result.success for result in group) < required_success:
            raise AssertionError(f"{track}: insufficient perfect seeds")
        if mean(result.base_margin for result in group) <= 0.02:
            raise AssertionError(f"{track}: weak base margin")
        if mean(result.retained_margin for result in group) <= 0.02:
            raise AssertionError(f"{track}: weak retained margin")

        control_off = sum(result.control_ablation_correct for result in group)
        temporal_off = sum(result.temporal_ablation_correct for result in group)
        reversed_correct = sum(result.reversed_correct for result in group)
        if base - control_off < 0.15 * base_total:
            raise AssertionError(f"{track}: query-control contribution")
        if base - temporal_off < 0.15 * base_total:
            raise AssertionError(f"{track}: temporal chunk contribution")
        if base - reversed_correct < 0.15 * base_total:
            raise AssertionError(f"{track}: sequence-order contribution")
        if mean(result.base_chunks for result in group) < 0.85 * len(CONCEPTS):
            raise AssertionError(f"{track}: base word-like chunk recall")
        if mean(result.later_chunks for result in group) < 0.85 * len(LATER_CONCEPTS):
            raise AssertionError(f"{track}: later word-like chunk recall")

        if track == "bio_connectome_local":
            lesion = sum(result.warm_lesion_correct for result in group)
            if base - lesion < 0.15 * base_total:
                raise AssertionError(
                    "bio_connectome_local: consolidation contribution"
                )

    audit = BioLocalSyllableDialogue.locality_audit().as_dict()
    required_local = {
        "boundary_free_syllable_input",
        "local_temporal_synaptic_update",
        "local_semantic_coactivity_update",
        "local_output_pre_post_update",
        "query_control_from_learned_temporal_activity",
        "no_autograd_or_weight_transport",
    }
    required_scaffolds = {
        "global_teacher_target",
        "global_window_enumeration",
        "global_sparse_activity_competition",
        "global_seed_balancing",
        "global_output_argmax",
    }
    if not all(audit[item] for item in required_local | required_scaffolds):
        raise AssertionError("incomplete locality audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    probe_results = run_probe(arguments.seeds, verbose=not arguments.quiet)
    if arguments.verify:
        verify_goal(probe_results)
