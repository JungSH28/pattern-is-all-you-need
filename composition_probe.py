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
    omit: str | None = None,
    contrast: str = "local",
) -> ConnectomeCompositionDialogue:
    model = ConnectomeCompositionDialogue(seed=seed, contrast=contrast)
    model.register_syllable_vocabulary(composition_vocabulary_texts())
    model.register_syllable_output(composition_vocabulary_texts())
    model.observe_streams(stage_streams(concepts, LABELED))
    model.observe_streams(tuple(item.question for item in training_pairs()))
    model.finalize_chunks()
    for entity, episode in zip(concepts, fact_episodes(concepts)):
        # Removing one composition target: this entity's facts are never
        # learned, so no assembly exists to co-activate.
        if entity == omit:
            continue
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


def result(seed: int, *, contrast: str = "local") -> dict[str, object]:
    model = _train(seed, contrast=contrast)
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
        # Single-fact baseline: each entity's own property, retrieved the #32
        # way. If these pass, the facts a composition would draw on are stored
        # and reachable, so a pair failure is composition and not memory.
        "single_fact_retrieval": sum(
            answer in model.generate(question)
            for question, answer in single_fact_questions()
        ),
        "single_fact_total": len(single_fact_questions()),
        "samples": tuple(
            (item.question, text or "(empty)", item.sentence)
            for item, text in zip(examples[:3], generated[:3])
        ),
    }


def partner_removed(seed: int) -> tuple[str, str]:
    """Ablate one composition target on a working (animal-containing) pair.

    Normal: wolf and truck are different categories, so composition answers
    that there is nothing in common. With truck's facts never learned, only
    wolf's assembly can co-activate, and a retrieval fallback claims wolf's own
    property -- the pair answer collapses to a single-fact answer.
    """
    question = PAIR_HELDOUT.format(a=SURFACE["wolf"], b=SURFACE["truck"])
    normal = _train(seed).generate(question)
    removed = _train(seed, omit="truck").generate(question)
    return normal, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--partner-seeds", type=int, default=3)
    arguments = parser.parse_args()

    # The contribution accounting: the local-contrast regime this goal adds,
    # the fixed top-k it must match without a global rank, and the bare
    # homeostatic threshold with no contrast at all.
    print(f"\n=== composition activity-regime ablation ({arguments.seeds} seeds) ===")
    print(f"  {'regime':18s} {'same':>7} {'  animal':>8} {' vehicle':>8} {'cross':>7} {'retr/cross':>11}")
    for regime in ("local", "topk", "none"):
        rows = [result(seed, contrast=regime) for seed in range(arguments.seeds)]
        same = sum(int(r["same_category"]) for r in rows)
        same_t = sum(int(r["same_total"]) for r in rows)
        animal = sum(
            judge(item, _train(seed, contrast=regime).generate(item.question))
            for seed in range(arguments.seeds)
            for item in heldout_pairs()
            if item.shared == "털"
        )
        animal_t = arguments.seeds * sum(1 for i in heldout_pairs() if i.shared == "털")
        vehicle = sum(
            judge(item, _train(seed, contrast=regime).generate(item.question))
            for seed in range(arguments.seeds)
            for item in heldout_pairs()
            if item.shared == "금속"
        )
        vehicle_t = arguments.seeds * sum(1 for i in heldout_pairs() if i.shared == "금속")
        cross = sum(int(r["cross_category"]) for r in rows)
        cross_t = sum(int(r["cross_total"]) for r in rows)
        retr = sum(int(r["retrieval_answer_on_cross_pairs"]) for r in rows)
        label = {"local": "local contrast", "topk": "fixed top-k", "none": "no contrast"}[regime]
        print(
            f"  {label:18s} {same:3d}/{same_t:<3d} {animal:4d}/{animal_t:<3d}"
            f" {vehicle:4d}/{vehicle_t:<3d} {cross:3d}/{cross_t:<3d} {retr:5d}/{cross_t}"
        )

    print("\n  ablation -- remove one composition target (wolf+truck, cross):")
    collapsed = 0
    for seed in range(arguments.partner_seeds):
        normal, removed = partner_removed(seed)
        # A cross pair collapses when dropping the partner makes the model claim
        # a shared property instead of answering "none".
        if "없어" in normal and "없어" not in removed:
            collapsed += 1
        print(f"    seed{seed}: both facts -> {normal} | partner removed -> {removed}")
    print(f"    collapsed to single-fact answer: {collapsed}/{arguments.partner_seeds}")


if __name__ == "__main__":
    main()
