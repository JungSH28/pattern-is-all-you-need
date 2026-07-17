"""Two-fact composition probe: what do these two things have in common?

No pair has a stored answer. A same-category pair shares a property; a
cross-category pair shares nothing and the answer is that there is nothing.
Held-out entities were never given a category and never appear in a training
pair, so a right answer cannot come from retrieving anything about them.

Category names, property sets and which pairs share what live only here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping, Sequence

from category_generalization_probe import CONCEPTS, HELD_OUT, LABELED
from composition_dialogue import ConnectomeCompositionDialogue
from syllable_chunk_probe import (
    SURFACE,
    fact_episodes,
    stage_streams,
)
from syllable_sentence_probe import sentence_vocabulary_texts

SHARED = {"animal": "털", "vehicle": "금속"}
CATEGORY_OF = {
    **{entity: "animal" for entity in ("cat", "dog", "horse", "wolf", "fox")},
    **{entity: "vehicle" for entity in ("car", "bus", "van", "truck", "bike")},
}

COMMON_SENTENCE = "둘다{content}이있어"
NONE_SENTENCE = "공통점은없어"

PAIR_TRAIN = (
    "{a}와{b}의공통점은뭐야",
    "{a}와{b}는뭐가같아",
)
# Neither wording occurs in training.
PAIR_HELDOUT = "{a}랑{b}가함께가진건뭘까"


@dataclass(frozen=True)
class PairExample:
    question: str
    sentence: str
    shared: str | None


def _pair(a: str, b: str, template: str) -> PairExample:
    same = CATEGORY_OF[a] == CATEGORY_OF[b]
    content = SHARED[CATEGORY_OF[a]] if same else None
    return PairExample(
        template.format(a=SURFACE[a], b=SURFACE[b]),
        COMMON_SENTENCE.format(content=content) if same else NONE_SENTENCE,
        content,
    )


def training_pairs() -> tuple[PairExample, ...]:
    prototypes = [e for entities in LABELED.values() for e in entities]
    built = []
    for template in PAIR_TRAIN:
        for index, a in enumerate(prototypes):
            for b in prototypes[index + 1 :]:
                built.append(_pair(a, b, template))
    return tuple(built)


def heldout_pairs() -> tuple[PairExample, ...]:
    animals = [e for e in HELD_OUT if CATEGORY_OF[e] == "animal"]
    vehicles = [e for e in HELD_OUT if CATEGORY_OF[e] == "vehicle"]
    built = [_pair(animals[0], animals[1], PAIR_HELDOUT)]
    built.append(_pair(vehicles[0], vehicles[1], PAIR_HELDOUT))
    for animal in animals:
        for vehicle in vehicles:
            built.append(_pair(animal, vehicle, PAIR_HELDOUT))
    return tuple(built)


def single_fact_questions() -> tuple[tuple[str, str], ...]:
    """Each entity's own property, asked the way #32 asks it.

    This is the retrieval baseline: it shows the facts a composing answer would
    have to draw on are individually stored and reachable. If these pass while
    the pair questions fail, the failure is composition, not memory.
    """
    return tuple(
        (
            "{entity}에게있는대표특징을말해".format(entity=SURFACE[entity]),
            SHARED[CATEGORY_OF[entity]],
        )
        for entity in HELD_OUT
    )


def composition_vocabulary_texts() -> tuple[str, ...]:
    return (
        *sentence_vocabulary_texts(),
        *(item.question for item in training_pairs()),
        *(item.sentence for item in training_pairs()),
        *(item.question for item in heldout_pairs()),
        *(item.sentence for item in heldout_pairs()),
        *(question for question, _ in single_fact_questions()),
        NONE_SENTENCE,
    )


def judge(example: PairExample, generated: str) -> bool:
    if example.shared is None:
        return "없어" in generated
    return example.shared in generated and "없어" not in generated


def _train(
    seed: int,
    *,
    concept_rounds: int = 30,
    pair_rounds: int = 6,
    concepts: Mapping[str, Sequence[str]] = CONCEPTS,
) -> ConnectomeCompositionDialogue:
    model = ConnectomeCompositionDialogue(seed=seed)
    model.register_syllable_vocabulary(composition_vocabulary_texts())
    model.register_syllable_output(composition_vocabulary_texts())
    model.observe_streams(stage_streams(concepts, LABELED))
    model.observe_streams(tuple(item.question for item in training_pairs()))
    model.finalize_chunks()
    for episode in fact_episodes(concepts):
        model.observe_fact_episode(episode, rounds=concept_rounds)
    same = [item for item in training_pairs() if item.shared is not None]
    cross = [item for item in training_pairs() if item.shared is None]
    # Six same-category pairs against nine cross pairs teaches "nothing in
    # common" as the safe answer. Balance the schedule, not the data.
    schedule = same * 2 + cross
    for _ in range(pair_rounds):
        for item in schedule:
            model.teach_sentence(item.question, item.sentence)
    model.consolidate_and_clear()
    return model


def result(seed: int) -> dict[str, object]:
    model = _train(seed)
    examples = heldout_pairs()
    generated = [model.generate(item.question) for item in examples]
    correct = [judge(item, text) for item, text in zip(examples, generated)]
    same = [item for item in examples if item.shared is not None]
    cross = [item for item in examples if item.shared is None]

    # A retrieval-only strategy answers a cross pair from the first entity
    # alone and so claims that entity's category property. Count how often the
    # model does that instead of composing.
    retrieval = sum(
        SHARED[CATEGORY_OF[entity]] in text
        for item, text in zip(examples, generated)
        if item.shared is None
        for entity in (
            [e for e in HELD_OUT if SURFACE[e] in item.question][:1]
        )
    )
    return {
        "seed": seed,
        "correct": sum(correct),
        "total": len(examples),
        "retrieval_answer_on_cross_pairs": retrieval,
        "retrieval_total": len(cross),
        "same_category": sum(
            judge(item, text)
            for item, text in zip(examples, generated)
            if item.shared is not None
        ),
        "same_total": len(same),
        "cross_category": sum(
            judge(item, text)
            for item, text in zip(examples, generated)
            if item.shared is None
        ),
        "cross_total": len(cross),

        "samples": tuple(
            (item.question, text or "(empty)", item.sentence)
            for item, text in zip(examples[:3], generated[:3])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    arguments = parser.parse_args()
    results = [result(seed) for seed in range(arguments.seeds)]
    print(f"\n=== composition, bio single connectome ({arguments.seeds} seeds) ===")
    for key, total in (
        ("correct", "total"),
        ("same_category", "same_total"),
        ("cross_category", "cross_total"),
        ("retrieval_answer_on_cross_pairs", "retrieval_total"),
    ):
        got = sum(int(item[key]) for item in results)
        out = sum(int(item[total]) for item in results)
        print(f"  {key:36s} {got:3d}/{out}")
    print("  samples (question -> generated | target):")
    for question, generated, target in results[0]["samples"]:
        print(f"    {question} -> {generated} | {target}")


if __name__ == "__main__":
    main()
