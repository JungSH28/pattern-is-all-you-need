"""Boundary-free syllable question to answer-sentence probe.

The answer is an ordered syllable emission, not one O-token.  Entity spans,
content spans and sentence frames exist only in this evaluator; the model API
receives the question text and the target sentence and nothing else.

Success is split so that a fluent frame with wrong content cannot be counted
as a success (the #19 generic-empathizer failure mode).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

from category_generalization_probe import CONCEPTS, HELD_OUT, LABELED
from syllable_chunk_dialogue import syllable_sequence
from syllable_chunk_probe import (
    CATEGORY_OUTPUT,
    FEATURE_OUTPUT,
    FEATURE_PROPERTY,
    SURFACE,
    fact_episodes,
    fact_records,
    fixed_syllable_vocabulary_texts,
    stage_streams,
    training_questions,
)
from syllable_sentence_dialogue import (
    ConnectomeSentenceDialogue,
    FunctionalSentenceDialogue,
)

CATEGORY_SENTENCE = "{entity}는{content}이야"
FEATURE_SENTENCE = "{entity}는{content}이있어"

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
CATEGORY_HELDOUT = "{entity}가속한종류를말해"
FEATURE_HELDOUT = "{entity}에게있는대표특징을말해"


@dataclass(frozen=True)
class SentenceExample:
    question: str
    sentence: str
    entity: str
    content: str


def _examples(
    entities: Mapping[str, str], *, held_out: bool
) -> tuple[SentenceExample, ...]:
    category_template = CATEGORY_HELDOUT if held_out else CATEGORY_TRAIN[0]
    feature_template = FEATURE_HELDOUT if held_out else FEATURE_TRAIN[0]
    built = []
    for entity, category in entities.items():
        surface = SURFACE[entity]
        category_word = CATEGORY_OUTPUT[category]
        feature_word = FEATURE_OUTPUT[FEATURE_PROPERTY[category]]
        built.append(
            SentenceExample(
                category_template.format(entity=surface),
                CATEGORY_SENTENCE.format(entity=surface, content=category_word),
                surface,
                category_word,
            )
        )
        built.append(
            SentenceExample(
                feature_template.format(entity=surface),
                FEATURE_SENTENCE.format(entity=surface, content=feature_word),
                surface,
                feature_word,
            )
        )
    return tuple(built)


def training_sentences(
    labeled: Mapping[str, Sequence[str]],
) -> tuple[SentenceExample, ...]:
    built = []
    for category, entities in labeled.items():
        category_word = CATEGORY_OUTPUT[category]
        feature_word = FEATURE_OUTPUT[FEATURE_PROPERTY[category]]
        for entity in entities:
            surface = SURFACE[entity]
            for template in CATEGORY_TRAIN:
                built.append(
                    SentenceExample(
                        template.format(entity=surface),
                        CATEGORY_SENTENCE.format(
                            entity=surface, content=category_word
                        ),
                        surface,
                        category_word,
                    )
                )
            for template in FEATURE_TRAIN:
                built.append(
                    SentenceExample(
                        template.format(entity=surface),
                        FEATURE_SENTENCE.format(
                            entity=surface, content=feature_word
                        ),
                        surface,
                        feature_word,
                    )
                )
    return tuple(built)


def heldout_sentences() -> tuple[SentenceExample, ...]:
    return _examples(HELD_OUT, held_out=True)


def sentence_vocabulary_texts() -> tuple[str, ...]:
    return (
        *fixed_syllable_vocabulary_texts(),
        *(item.sentence for item in training_sentences(LABELED)),
        *(item.sentence for item in heldout_sentences()),
    )


def _frame(example: SentenceExample) -> tuple[str, ...]:
    """Sentence syllables outside the entity and content spans."""
    answer = syllable_sequence(example.sentence)
    entity = syllable_sequence(example.entity)
    content = syllable_sequence(example.content)
    frame: list[str] = []
    index = 0
    while index < len(answer):
        if tuple(answer[index : index + len(entity)]) == entity:
            index += len(entity)
        elif tuple(answer[index : index + len(content)]) == content:
            index += len(content)
        else:
            frame.append(answer[index])
            index += 1
    return tuple(frame)


def _ordered_subsequence(whole: Sequence[str], part: Sequence[str]) -> bool:
    iterator = iter(whole)
    return all(any(item == token for item in iterator) for token in part)


@dataclass(frozen=True)
class SentenceOutcome:
    exact: bool
    entity_ok: bool
    content_ok: bool
    frame_ok: bool

    @property
    def form_only(self) -> bool:
        """Frame reproduced but the content is wrong: the #19 failure mode."""
        return self.frame_ok and not self.content_ok


def judge(example: SentenceExample, generated: str) -> SentenceOutcome:
    produced = syllable_sequence(generated)
    return SentenceOutcome(
        exact=generated == example.sentence,
        entity_ok=example.entity in generated,
        content_ok=example.content in generated,
        frame_ok=_ordered_subsequence(produced, _frame(example)),
    )


def report(
    outcomes: Sequence[SentenceOutcome],
) -> dict[str, int]:
    return {
        "exact": sum(item.exact for item in outcomes),
        "entity": sum(item.entity_ok for item in outcomes),
        "content": sum(item.content_ok for item in outcomes),
        "frame": sum(item.frame_ok for item in outcomes),
        "form_only": sum(item.form_only for item in outcomes),
        "total": len(outcomes),
    }


def functional_result(seed: int) -> dict[str, object]:
    model = FunctionalSentenceDialogue(seed=seed)
    model.register_syllable_vocabulary(sentence_vocabulary_texts())
    model.register_output_vocabulary(
        tuple(CATEGORY_OUTPUT.values()) + tuple(FEATURE_OUTPUT.values())
    )
    model.observe_streams(stage_streams(CONCEPTS, LABELED))
    model.finalize_chunks()
    for _ in range(16):
        for text, properties in fact_records(CONCEPTS):
            model.observe_fact(text, properties)
    for _ in range(16):
        for text, target in training_questions(LABELED):
            model.teach_answer(text, target)
    for item in training_sentences(LABELED):
        model.teach_sentence(item.question, item.sentence, item.content)

    outcomes = [
        judge(item, model.generate(item.question)) for item in heldout_sentences()
    ]
    samples = tuple(
        (item.question, model.generate(item.question), item.sentence)
        for item in heldout_sentences()[:3]
    )
    return {
        "track": "functional_global",
        "seed": seed,
        **report(outcomes),
        "perfect": int(all(item.exact for item in outcomes)),
        "samples": samples,
    }


def _trained_bio(
    seed: int, *, concept_rounds: int, sentence_rounds: int, **flags
) -> ConnectomeSentenceDialogue:
    model = ConnectomeSentenceDialogue(seed=seed, **flags)
    model.register_syllable_vocabulary(sentence_vocabulary_texts())
    model.register_syllable_output(sentence_vocabulary_texts())
    model.observe_streams(stage_streams(CONCEPTS, LABELED))
    model.finalize_chunks()
    for episode in fact_episodes(CONCEPTS):
        model.observe_fact_episode(episode, rounds=concept_rounds)
    for _ in range(sentence_rounds):
        for item in training_sentences(LABELED):
            model.teach_sentence(item.question, item.sentence)
    model.consolidate_and_clear()
    return model


def bio_result(
    seed: int, *, concept_rounds: int = 30, sentence_rounds: int = 4
) -> dict[str, object]:
    build = lambda **flags: _trained_bio(
        seed,
        concept_rounds=concept_rounds,
        sentence_rounds=sentence_rounds,
        **flags,
    )
    model = build()
    generated = [model.generate(item.question) for item in heldout_sentences()]
    outcomes = [
        judge(item, text) for item, text in zip(heldout_sentences(), generated)
    ]
    # The name replay, the query gate and the order stream are ablated one at a
    # time. Each is trained and tested in its ablated form rather than being
    # switched off only at generation, so a control cannot coast on branches
    # that the intact mechanism recruited.
    no_replay = build(use_replay=False)
    no_trace = build(use_trace=False)
    ablations = {
        "no_replay": sum(
            judge(item, no_replay.generate(item.question)).entity_ok
            for item in heldout_sentences()
        ),
        "no_trace": sum(
            judge(item, no_trace.generate(item.question)).exact
            for item in heldout_sentences()
        ),
        "no_control": sum(
            judge(item, model.generate(item.question, controlled=False)).content_ok
            for item in heldout_sentences()
        ),
    }
    return {
        "track": "bio_single_connectome",
        "seed": seed,
        **report(outcomes),
        **ablations,
        "perfect": int(all(item.exact for item in outcomes)),
        "branches": model.connectome.dendritic_branch_count(),
        "samples": tuple(
            (item.question, text, item.sentence)
            for item, text in zip(heldout_sentences()[:3], generated[:3])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--track", choices=("functional", "bio", "both"), default="both")
    arguments = parser.parse_args()

    tracks = []
    if arguments.track in ("functional", "both"):
        tracks.append(functional_result)
    if arguments.track in ("bio", "both"):
        tracks.append(bio_result)

    for runner in tracks:
        results = [runner(seed) for seed in range(arguments.seeds)]
        first = results[0]
        print(f"\n=== {first['track']} ({arguments.seeds} seeds) ===", flush=True)
        keys = ["exact", "entity", "content", "frame", "form_only"]
        keys += [key for key in ("no_replay", "no_trace", "no_control") if key in first]
        for key in keys:
            total = sum(int(item[key]) for item in results)
            denominator = sum(int(item["total"]) for item in results)
            print(f"  {key:10s} {total:3d}/{denominator}")
        perfect = sum(int(item["perfect"]) for item in results)
        print(f"  {'perfect':10s} {perfect:3d}/{len(results)} seeds")
        if "branches" in first:
            print(f"  branches   {mean(float(item['branches']) for item in results):.0f}")
        print("  samples (question -> generated | target):")
        for question, generated, target in first["samples"]:
            print(f"    {question} -> {generated or '(empty)'} | {target}")


if __name__ == "__main__":
    main()
