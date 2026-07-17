"""Structure tests for answer-sentence generation.

These pin the mechanisms the sentence track claims, and the API boundary: no
entity span, content span, sentence frame or answer candidate may reach the
model.
"""

from __future__ import annotations

import inspect
import unittest

from category_generalization_probe import CONCEPTS, LABELED
from syllable_chunk_probe import fact_episodes, stage_streams
from syllable_sentence_dialogue import ConnectomeSentenceDialogue
from syllable_sentence_probe import (
    heldout_sentences,
    sentence_vocabulary_texts,
    training_sentences,
)


def _grounded(seed: int = 0) -> ConnectomeSentenceDialogue:
    model = ConnectomeSentenceDialogue(seed=seed)
    model.register_syllable_vocabulary(sentence_vocabulary_texts())
    model.register_syllable_output(sentence_vocabulary_texts())
    model.observe_streams(stage_streams(CONCEPTS, LABELED))
    model.finalize_chunks()
    for episode in fact_episodes(CONCEPTS):
        model.observe_fact_episode(episode, rounds=30)
    return model


class SentenceApiBoundary(unittest.TestCase):
    def test_public_api_takes_only_text_and_target_sentence(self):
        signature = inspect.signature(ConnectomeSentenceDialogue.teach_sentence)
        self.assertEqual(
            [name for name in signature.parameters if name != "self"],
            ["text", "sentence", "learn_control"],
        )
        generate = inspect.signature(ConnectomeSentenceDialogue.generate)
        self.assertEqual(
            [name for name in generate.parameters if name != "self"],
            ["text", "controlled"],
        )


class NameReplay(unittest.TestCase):
    """The open-class name has to come from the entity's own chunk."""

    def setUp(self):
        self.model = _grounded()

    def test_chunk_drives_its_own_syllables_above_everything_else(self):
        for entity in self.model.concept_patterns:
            drive = self.model._constituent_drive(entity)
            weakest = min(drive[syllable] for syllable in entity)
            strongest_other = max(
                value
                for syllable, value in drive.items()
                if syllable not in entity
            )
            self.assertGreater(weakest, strongest_other, msg=f"{entity}")

    def test_replay_speaks_the_name_in_order(self):
        for entity in self.model.concept_patterns:
            spoken: list[str] = []
            while len(spoken) <= len(entity):
                syllable = self.model._replay_syllable(entity, spoken)
                if syllable is None:
                    break
                spoken.append(syllable)
            self.assertEqual(tuple(spoken), entity)

    def test_held_out_names_replay_without_any_sentence_target(self):
        taught = {item.entity for item in training_sentences(LABELED)}
        for item in heldout_sentences():
            self.assertNotIn(item.entity, taught)
            entity = self.model._entity_pattern(item.question)
            spoken: list[str] = []
            while len(spoken) <= len(entity):
                syllable = self.model._replay_syllable(entity, spoken)
                if syllable is None:
                    break
                spoken.append(syllable)
            self.assertEqual("".join(spoken), item.entity)


class TwoStreamBranches(unittest.TestCase):
    def test_streams_are_normalized_apart(self):
        """Meaning and order must not compete for one normalization budget."""
        model = _grounded()
        for _ in range(4):
            for item in training_sentences(LABELED):
                model.teach_sentence(item.question, item.sentence)
        model.consolidate_and_clear()

        question = heldout_sentences()[0].question
        entity = model._entity_pattern(question)
        spoken = list(entity)
        emitted: list = []
        context = model._emission_context(question, emitted)
        first = model.connectome.dendritic_sequence_emission(context)
        token, (_, order) = max(first.items(), key=lambda item: item[1][0])
        self.assertGreaterEqual(order, 0.65)

        # Once the trace no longer matches any recruited context the order
        # stream collapses; that is what ends the sentence.
        emitted = [
            model.connectome.output_pattern(syllable) for syllable in ("이", "있", "어")
        ]
        stale = model._emission_context(question, emitted)
        _, ended = model._branch_syllable(stale)
        self.assertLess(ended, 0.65)

    def test_replay_output_never_enters_the_trace(self):
        source = inspect.getsource(ConnectomeSentenceDialogue.generate)
        self.assertIn('if source != "replay"', source)


if __name__ == "__main__":
    unittest.main()
