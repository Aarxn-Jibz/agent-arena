import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from arena.training import (Checkpoints, Curriculum, EpisodeOrchestrator, HFPEFTTrainer, MockTrainer, ModelConfig,
                            SolverResponse, WhitespaceTokenizer, challenger_reward, compose_context,
                            doctor_recommendations, failure_feedback, normalize_resource_status,
                            outcome_key, parse_solver_contract, reference_identity, validate_manifests,
                            verified_correction, select_judge, WslJudge, WindowsNativeJudge, estimate_storage)


class TrainingArchitectureTests(unittest.TestCase):
    def test_chat_template_text_is_tokenized_before_context_check(self):
        class Tensor:
            def __init__(self, values): self.values, self.device = values, None
            @property
            def shape(self): return (1, len(self.values))
            def to(self, device): self.device = device; return self
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): self.template = (messages, kwargs); return 'rendered chat prompt'
            def __call__(self, text, **kwargs): self.tokenized = (text, kwargs); return {'input_ids': Tensor([1, 2, 3]), 'attention_mask': Tensor([1, 1, 1])}
        class Parameter: device = 'cpu'
        class Model:
            def parameters(self): return iter([Parameter()])
        class Torch:
            pass
        Torch.Tensor = Tensor
        trainer = HFPEFTTrainer(ModelConfig(), dependencies={'torch': Torch()})
        trainer.model, trainer.tokenizer = Model(), Tokenizer()
        inputs = trainer._generation_inputs('prompt')
        self.assertEqual(inputs['input_ids'].shape[1], 3)
        self.assertEqual(inputs['input_ids'].device, 'cpu')
        self.assertEqual(inputs['attention_mask'].device, 'cpu')
        self.assertEqual(trainer.tokenizer.tokenized[0], 'rendered chat prompt')
        self.assertFalse(trainer.tokenizer.template[1]['tokenize'])

    def test_generation_uses_chat_continuation_slice_and_code_stop(self):
        class Ids:
            def __init__(self, values, batched=False): self.values, self.batched = values, batched
            @property
            def shape(self): return (1, len(self.values)) if self.batched else (len(self.values),)
            def to(self, device): return self
            def __getitem__(self, key):
                if isinstance(key, tuple): return Ids(self.values[key[1]])
                return self.values[key]
        class Tokenizer:
            pad_token_id = eos_token_id = 0
            def apply_chat_template(self, messages, **kwargs): self.messages = messages; return 'rendered'
            def __call__(self, text, **kwargs): self.tokenized = text; return {'input_ids': Ids([10, 11], True)}
            def decode(self, ids, **kwargs):
                values = ids.values if isinstance(ids, Ids) else ids
                return '\\<CODE>\n<C>\n#include <stdio.h>\nint main(void){return 0;}\n\\</C> trailing' if values == [20, 21, 22] else '<CODE>one complete C program</CODE>'
        class Parameter: device = 'cpu'
        class Model:
            def parameters(self): return iter([Parameter()])
            def named_parameters(self): return []
            def set_adapter(self, role): pass
            def generate(self, **kwargs):
                self.kwargs = kwargs
                self.stopped = kwargs['stopping_criteria'][0](Ids([10, 11, 20, 21, 22], True), None)
                return Ids([10, 11, 20, 21, 22], True)
        class Torch:
            Tensor = Ids
            @staticmethod
            def inference_mode(): return nullcontext()
        trainer = HFPEFTTrainer(ModelConfig(context_length=3000), dependencies={'torch': Torch()})
        trainer.model, trainer.tokenizer = Model(), Tokenizer()
        result = trainer.generate('solver', '<CODE>one complete C program</CODE>', {'max_new_tokens': 2500})
        self.assertEqual(trainer.tokenizer.messages, [{'role': 'user', 'content': '<CODE>one complete C program</CODE>'}])
        self.assertEqual(trainer.model.kwargs['max_new_tokens'], 2500)
        self.assertTrue(trainer.model.stopped); self.assertEqual(result['tokens'], 3)
        self.assertEqual(result['text'], '\\<CODE>\n<C>\n#include <stdio.h>\nint main(void){return 0;}\n\\</C>')
        self.assertNotIn('one complete C program', result['text'])

    def test_hf_backend_adapter_isolation_checkpoint_and_offline_load(self):
        class Param:
            def __init__(self, n): self.n, self.requires_grad, self.device = n, True, "cpu"
            def numel(self): return self.n
        class Model:
            def __init__(self):
                self.params = {"base.weight": Param(100), "x.challenger.lora": Param(3), "x.solver.lora": Param(5)}; self.saved = []; self.loaded = []; self.saved_paths = []; self.loaded_paths = []
            def parameters(self): return self.params.values()
            def named_parameters(self): return self.params.items()
            def add_adapter(self, *x): pass
            def set_adapter(self, name): self.active = name
            def save_pretrained(self, path, selected_adapters):
                adapter = Path(path) / selected_adapters[0]; adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text("{}"); (adapter / "adapter_model.safetensors").write_text("weights")
                self.saved.append(selected_adapters[0]); self.saved_paths.append(path)
            def load_adapter(self, path, adapter_name, is_trainable):
                if not (Path(path) / "adapter_config.json").is_file() or not (Path(path) / "adapter_model.safetensors").is_file():
                    raise AssertionError("adapter artifacts missing")
                self.loaded.append(adapter_name); self.loaded_paths.append(path)
        class Tokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs): return Tokenizer()
        class Loader:
            @staticmethod
            def from_pretrained(*args, **kwargs): return Model()
        class Opt:
            def __init__(self, params, lr): self.params, self.lr = list(params), lr
            def state_dict(self): return {"lr": self.lr}
            def load_state_dict(self, value): self.restored = value
        class Torch:
            class optim: AdamW = Opt
            @staticmethod
            def save(value, path): Path(path).write_text(str(value))
            @staticmethod
            def load(path, **kwargs): return {"restored": True}
            bfloat16 = float; float16 = float; float32 = float
        class Config:
            def __init__(self, **kwargs): self.kwargs = kwargs
        deps = {"torch": Torch, "AutoTokenizer": Tokenizer, "AutoModelForCausalLM": Loader,
                "BitsAndBytesConfig": Config, "LoraConfig": Config, "TaskType": type("T", (), {"CAUSAL_LM": "causal"}),
                "get_peft_model": lambda model, config, adapter_name: model, "prepare_model_for_kbit_training": lambda model: model}
        trainer = HFPEFTTrainer(ModelConfig(lora={"r": 2, "alpha": 4}), dependencies=deps).load()
        self.assertEqual(trainer.active_adapter, "challenger")
        self.assertNotEqual(trainer.optimizers["challenger"], trainer.optimizers["solver"])
        self.assertEqual(trainer.adapter_state("solver")["trainable"], 5)
        calls = []; trainer._solver_loss = lambda example: 1; trainer._step = lambda role, loss: calls.append(role) or 1.0
        trainer.update_solver([{"input": {}, "target": {}}]); trainer.update_challenger({"valid": True, "log_probability": 1, "reward": 1})
        self.assertEqual(calls, ["solver", "challenger"])
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / "new"; trainer.save_checkpoint(checkpoint); trainer.load_checkpoint(checkpoint)
            self.assertEqual(trainer.model.saved, ["challenger", "solver"]); self.assertEqual(trainer.model.loaded, ["challenger", "solver"])
            self.assertTrue(all(isinstance(path, str) for path in trainer.model.saved_paths + trainer.model.loaded_paths))
            self.assertTrue(all(Path(path).parent == checkpoint for path in trainer.model.loaded_paths))
            legacy = Path(d) / "legacy"; shutil.copytree(checkpoint, legacy)
            for role in trainer.ROLES:
                adapter = legacy / role; nested = adapter / role; nested.mkdir()
                for item in list(adapter.iterdir()):
                    if item != nested: item.rename(nested / item.name)
            trainer.model.loaded = []; trainer.model.loaded_paths = []; trainer.load_checkpoint(legacy)
            self.assertEqual(trainer.model.loaded, ["challenger", "solver"])
            self.assertTrue(all(Path(path).parent.parent == legacy for path in trainer.model.loaded_paths))

    def test_local_adapter_path_rejects_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "local solver adapter checkpoint is incomplete"):
                HFPEFTTrainer._local_adapter_path(Path(d), "solver")

    def test_hf_backend_missing_libraries_fails_without_download(self):
        with self.assertRaisesRegex(RuntimeError, "HF/PEFT runtime requires"):
            HFPEFTTrainer(ModelConfig(), dependencies={}).load()

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
        self.assertIn("int main", parse_solver_contract("```c\nint main(){}\n```").code)
        self.assertEqual(reference_identity("one\n\ntwo", "v1")["section_ids"], ["R1", "R2"])

    def test_solver_contract_tolerates_tags_fences_and_prose(self):
        fenced = parse_solver_contract("<STRATEGY>short</STRATEGY><REFERENCE_USAGE>R1/R2</REFERENCE_USAGE><CODE>```c\nint main(void){return 0;}\n```</CODE>")
        prose = parse_solver_contract("note\n<CODE>int main(void){return 0;}</CODE>\ndone")
        legacy = parse_solver_contract('{"summary":"x","changes":[]}')
        fallback = parse_solver_contract('```c\n#include <stdio.h>\nint main(void){return 0;}\n```')
        self.assertEqual(fenced.reference_ids, ["R1", "R2"])
        self.assertEqual(fenced.code, "int main(void){return 0;}\n")
        self.assertEqual(prose.strategy, ""); self.assertIn("int main", prose.code)
        self.assertEqual(legacy.malformed, "missing CODE section")
        self.assertIn('int main', fallback.code)

    def test_solver_contract_recovers_escaped_nested_c_with_trailing_text(self):
        reply = r'\<CODE>' + '\n<C>\n#include <stdio.h>\nint main(void){return 1;}\n' + r'\</C>' + '\nrambling'
        parsed = parse_solver_contract(reply)
        self.assertFalse(parsed.malformed)
        self.assertEqual(parsed.code, '#include <stdio.h>\nint main(void){return 1;}\n')

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
