import inspect
import unittest

import torch

from syllable_chunk_dialogue import (
    BioLocalSyllableDialogue,
    BioLocalTemporalChunker,
    ConnectomeSyllableDialogue,
    FunctionalSyllableDialogue,
    FunctionalTemporalChunker,
    syllable_sequence,
)
from syllable_chunk_probe import bio_result, functional_result


class BoundaryFreeChunkTests(unittest.TestCase):
    def test_tokenizer_exposes_only_continuous_syllables(self):
        self.assertEqual(
            syllable_sequence("늑대는 무엇인가"),
            ("늑", "대", "는", "무", "엇", "인", "가"),
        )

    def test_functional_chunk_forms_only_after_repetition(self):
        chunker = FunctionalTemporalChunker(n_units=64, assembly_size=4, seed=2)
        chunker.observe("늑대는달린다")
        chunker.finalize()
        self.assertIsNone(chunker.pattern_code(("늑", "대")))
        chunker.observe("늑대가달린다")
        chunker.finalize()
        code = chunker.pattern_code(("늑", "대"))
        self.assertIsNotNone(code)
        self.assertEqual(int(code.sum().item()), 4)

    def test_bio_temporal_update_touches_only_active_pre_post_edges(self):
        chunker = BioLocalTemporalChunker(n_units=64, assembly_size=4, seed=3)
        chunker.register_syllables(("늑", "대", "여", "우"))
        before = chunker.transition.clone()
        chunker.observe("늑대")
        changed = chunker.transition != before
        pre = chunker.bank.code("늑") > 0
        post = chunker.bank.code("대") > 0
        expected = torch.outer(pre, post)
        self.assertTrue(torch.equal(changed, expected))

    def test_public_api_has_no_entity_query_or_candidate_argument(self):
        for model_type in (
            FunctionalSyllableDialogue,
            BioLocalSyllableDialogue,
            ConnectomeSyllableDialogue,
        ):
            answer = inspect.signature(model_type.answer).parameters
            teach = inspect.signature(model_type.teach_answer).parameters
            for forbidden in ("entity", "query", "intent", "candidates"):
                self.assertNotIn(forbidden, answer)
                self.assertNotIn(forbidden, teach)


class BoundaryFreeEndToEndTests(unittest.TestCase):
    def test_seed_zero_functional_path_is_end_to_end(self):
        self.assertTrue(functional_result(0).success)

    def test_seed_zero_bio_local_path_is_end_to_end(self):
        self.assertTrue(bio_result(0).success)

    def test_locality_audit_exposes_local_rules_and_scaffolds(self):
        audit = ConnectomeSyllableDialogue.locality_audit()
        self.assertTrue(audit.single_connectome_memory_and_output)
        self.assertTrue(audit.local_temporal_synaptic_update)
        self.assertTrue(audit.local_semantic_coactivity_update)
        self.assertTrue(audit.local_output_pre_post_update)
        self.assertTrue(audit.query_control_from_learned_temporal_activity)
        self.assertTrue(audit.local_query_control_prototype_update)
        self.assertTrue(audit.global_window_enumeration)
        self.assertTrue(audit.global_fact_episode_intersection)
        self.assertTrue(audit.global_sparse_activity_competition)
        self.assertTrue(audit.global_output_argmax)


if __name__ == "__main__":
    unittest.main()
