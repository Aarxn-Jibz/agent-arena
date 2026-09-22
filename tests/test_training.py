import tempfile
import unittest
from pathlib import Path

from arena.training import (Checkpoints, Curriculum, EpisodeOrchestrator, MockTrainer, ModelConfig,
                            SolverResponse, WhitespaceTokenizer, challenger_reward, compose_context,
                            doctor_recommendations, failure_feedback, normalize_resource_status,
                            outcome_key, parse_solver_contract, reference_identity, validate_manifests,
                            verified_correction, select_judge, WslJudge, WindowsNativeJudge, estimate_storage)


class TrainingArchitectureTests(unittest.TestCase):
    def test_doctor_profiles(self):
        self.assertEqual(doctor_recommendations({"cuda": False})["solver_attempts"], 2)
        self.assertEqual(doctor_recommendations({"cuda": True, "vram_gb": 16})["adaptation"], "QLoRA")
        self.assertEqual(doctor_recommendations({"cuda": True, "vram_gb": 24, "bf16": True})["solver_attempts"], 3)
        self.assertEqual(doctor_recommendations({"cuda": True, "vram_gb": 80, "bf16": True})["precision"], "bf16")

    def test_curriculum_progression_bounds_and_fallback(self):
        c = Curriculum(["compression"])
        self.assertFalse(c.validate({"benchmark": "compression", "difficulty": 8})[0])
        for _ in range(3): c.record("compression", 1, {"success": True, "fraction": 1, "compiled": True})
        self.assertEqual(c.progress["compression"].frontier, 2)
        self.assertTrue(c.validate({"benchmark": "compression", "difficulty": 2})[0])
        for _ in range(3): c.record("compression", 2, {"success": True, "fraction": 1, "compiled": True})
        self.assertEqual(c.progress["compression"].frontier, 3)
        c.record("compression", 3, {"success": False, "fraction": 0, "compiled": False})
        self.assertLessEqual(max(x["difficulty"] for x in c.legal_actions("compression")), 3)

    def test_challenger_reward_invalid_and_fallback_not_credited(self):
        self.assertLess(challenger_reward(False, False, {})["total"], 0)
        self.assertEqual(challenger_reward(True, True, {"fraction": .5})["total"], 0)
        self.assertGreater(challenger_reward(True, False, {"fraction": .5, "repair_success": True})["total"], 1)

    def test_solver_contract_reference_none_and_bad(self):
        good = parse_solver_contract("<STRATEGY>x</STRATEGY>\n<REFERENCE_USAGE>R2 - scanf details</REFERENCE_USAGE>\n<CODE>int main(){} </CODE>")
        self.assertEqual(good.reference_ids, ["R2"]); self.assertIn("int main", good.code)
        self.assertEqual(parse_solver_contract("<STRATEGY>x</STRATEGY><REFERENCE_USAGE>NONE</REFERENCE_USAGE><CODE>x</CODE>").reference_ids, [])
        self.assertIsNotNone(parse_solver_contract("```c\nint main(){}\n```").malformed)
        self.assertEqual(reference_identity("one\n\ntwo", "v1")["section_ids"], ["R1", "R2"])

    def test_resource_feedback_and_large_logs(self):
        self.assertEqual(normalize_resource_status(None)["status"], "unknown")
        self.assertEqual(normalize_resource_status("within limits")["status"], "within limits")
        self.assertEqual(normalize_resource_status({"exit_code": 1})["exit_code"], 1)
        fb = failure_feedback("task", SolverResponse("s", [], {}, "C"), {"build": {"exit_code": 1, "stderr": "x" * 5000}, "performance": {"resource_status": "timeout"}}, {})
        self.assertTrue(fb["compilation"]["diagnostics"]["truncated"]); self.assertEqual(fb["resource"]["status"], "timeout")

    def test_context_order_and_correction(self):
        cfg = ModelConfig(context_length=12, generation={"max_new_tokens": 4})
        _, meta = compose_context({"system": "a b", "task": "c d", "challenge": "e f", "reference": "g h", "history": "i j"}, cfg, WhitespaceTokenizer())
        self.assertIn("history", meta["removed"]); self.assertEqual(meta["reserved_output_tokens"], 4)
        bad = SolverResponse("bad", [], {}, "bad"); good = SolverResponse("good", [], {}, "good")
        correction = verified_correction([(bad, {"compiled": False}, {"why": "no"}), (good, {"compiled": True, "passed": 1, "total": 1}, {})])
        self.assertEqual(correction["target"]["code"], "good")
        self.assertGreater(outcome_key({"compiled": True, "passed": 1, "total": 1}), outcome_key({"compiled": False}))

    def test_orchestration_idempotency_evaluation_and_retention(self):
        with tempfile.TemporaryDirectory() as d:
            c, t, cp = Curriculum(["csv"]), MockTrainer(), Checkpoints(Path(d), milestones=1)
            o = EpisodeOrchestrator(t, c, cp)
            a = {"benchmark": "csv", "difficulty": 1}
            attempts = [(SolverResponse("a", [], {}, "a"), {"compiled": False}, {}), (SolverResponse("b", [], {}, "b"), {"compiled": True, "passed": 1, "total": 1}, {})]
            run = {"run_id": "x"}; first = o.commit(run, 1, "csv", a, attempts, {})
            self.assertTrue(first["updated"]); self.assertEqual(len(t.updates), 2)
            self.assertTrue(o.commit(run, 1, "csv", a, attempts, {})["duplicate"]); self.assertEqual(len(t.updates), 2)
            e = EpisodeOrchestrator(t, c, cp, evaluation=True); e.commit(run, 2, "csv", a, attempts, {})
            self.assertEqual(len(t.updates), 2); self.assertTrue((Path(d) / "latest/state.json").exists())

    def test_isolation_judges_storage_and_leakage(self):
        with self.assertRaises(ValueError): validate_manifests([{"id": 1}], [{"id": 1}])
        validate_manifests([{"id": 1}], [{"id": 2}])
        self.assertTrue(callable(select_judge("docker"))); self.assertIsInstance(select_judge("wsl"), WslJudge); self.assertIsInstance(select_judge("windows-native"), WindowsNativeJudge)
        self.assertEqual(estimate_storage(11, 10, 100, 10), 310)
