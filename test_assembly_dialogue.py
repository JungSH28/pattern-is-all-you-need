import unittest

from assembly_dialogue import BioLocalAssemblyDialogue, FunctionalAssemblyDialogue


class FunctionalAssemblyDialogueTests(unittest.TestCase):
    def test_same_entity_answers_different_queries_from_full_vocabulary(self):
        model = FunctionalAssemblyDialogue(seed=3)
        model.register_queries(("what_is", "what_feature"))
        model.register_vocabulary(("unknown", "yes", "no"))
        model.observe_fact("cat", ("fur", "fourlegs"))
        model.teach_answer("what_is", "cat", "animal")
        model.teach_answer("what_feature", "cat", "fur")

        category, _, category_scores = model.answer("what_is", "cat")
        feature, _, feature_scores = model.answer("what_feature", "cat")
        self.assertEqual(category, "animal")
        self.assertEqual(feature, "fur")
        self.assertEqual(set(category_scores), set(model.codes))
        self.assertEqual(set(feature_scores), set(model.codes))

    def test_token_codes_are_sparse_unit_sets(self):
        model = FunctionalAssemblyDialogue(n_units=64, assembly_size=4, seed=2)
        model.register_vocabulary(("cat", "dog", "animal"))
        for code in model.codes.values():
            self.assertEqual(int(code.sum().item()), 4)


class BioLocalAuditTests(unittest.TestCase):
    def test_locality_audit_exposes_remaining_scaffolds(self):
        audit = BioLocalAssemblyDialogue.locality_audit()
        self.assertTrue(audit.local_synaptic_value_update)
        self.assertTrue(audit.source_local_stochastic_rewiring)
        self.assertFalse(audit.global_structural_search)
        self.assertTrue(audit.global_activity_competition)
        self.assertTrue(audit.global_seed_and_gate_balancing)


if __name__ == "__main__":
    unittest.main()
