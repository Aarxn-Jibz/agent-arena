import json
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arena import experiment
from arena.training import MockTrainer
from arena.experiment_context import apply_changes, lexical_search, parse_solver_response, read_range
from arena.sandbox import SandboxConfig


class FakeBenchmark:
    name = 'compression'

    def initialize(self, seed):
        return {'seed': seed, 'size': 1}

    def validate_challenge(self, value):
        return (set(value) == {'seed', 'size'} and value['size'] == 1, 'invalid')

    def evaluate(self, files, challenge, config):
        text = files.get('solution.c', '')
        passed = 'return 0' in text
        return {'started_at': experiment.now(), 'finished_at': experiment.now(),
                'accepted': passed, 'build': {'exit_code': 0, 'stdout': '', 'stderr': ''},
                'correctness': {'passed': int(passed), 'total': 1, 'cases': []},
                'performance': {'median_ms': 100 if 'version2' not in text else 80},
                'reward_inputs': {'solver_reward': float(passed), 'challenger_reward': float(not passed)},
                'feedback': '1/1 passed' if passed else '0/1 passed'}


class FakeModel:
    revision = 'test-revision'

    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, seed, max_new_tokens):
        self.calls += 1
        if 'Challenger' in messages[0]['content']:
            data = {'challenge': {'seed': seed, 'size': 1}, 'rationale': 'probe correctness'}
        else:
            code = 'int main(void){return 0;} /* version2 */' if seed > 501000 else 'int main(void){return 0;}'
            data = {'summary': 'small candidate', 'changes': [{'path': 'solution.c', 'content': code}]}
        return {'text': json.dumps(data), 'tokens': 30, 'truncated': False, 'prompt_tokens': 100}


class ExperimentTests(unittest.TestCase):
    def test_eval_manifest_benchmark_outside_training_counts_correctly(self):
        class HeldOutBenchmark:
            name = 'cache'
            def evaluate(self, source, challenge, config):
                return {'started_at': experiment.now(), 'finished_at': experiment.now(), 'accepted': False,
                        'build': {'exit_code': 0, 'stdout': '', 'stderr': ''},
                        'correctness': {'passed': 0, 'total': 1, 'cases': []}, 'performance': {'median_ms': 1},
                        'reward_inputs': {'solver_reward': 0.0, 'challenger_reward': 1.0}, 'feedback': 'rejected'}
        class EvalModel:
            revision = 'test-revision'
            def generate(self, messages, **kwargs):
                return {'text': '{"summary":"x","changes":[{"path":"solution.c","content":"int main(void){return 0;}"}]}'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repo'; root.mkdir(); subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'README').write_text('x')
            for name in ('cache', 'log', 'search'):
                path = root / 'docs' / 'benchmarks' / name / 'SPEC.md'; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('spec')
            env = os.environ | {'GIT_AUTHOR_NAME': 'Test', 'GIT_AUTHOR_EMAIL': 'test@example.com',
                                'GIT_COMMITTER_NAME': 'Test', 'GIT_COMMITTER_EMAIL': 'test@example.com'}
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True); subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'initial'], check=True, env=env)
            manifest_path = Path(temp) / 'eval.json'
            judge = {'cpus': 1, 'memory_mb': 256, 'pids': 64, 'tmpfs_mb': 128, 'timeout_seconds': 3, 'output_bytes': 65536}
            with patch.object(experiment, 'ROOT', root):
                manifest = experiment.create_eval_manifest(manifest_path, 42, 1, judge)
                manifest['entries'] = [manifest['entries'][0]]
                manifest['sha256'] = __import__('hashlib').sha256(json.dumps({key: value for key, value in manifest.items() if key != 'sha256'}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                experiment.atomic_json(manifest_path, manifest)
                args = SimpleNamespace(root=Path(temp) / 'runs', run_id='eval_count', seed=42, hours=None, episodes=None,
                    resume=False, benchmarks='compression', cpus=1, selection_mode='round_robin', memory_mb=256, pids=64,
                    tmpfs_mb=128, timeout_seconds=3, output_bytes=65536, soft_gb=.1, hard_gb=.2, challenger_tokens=10,
                    solver_tokens=50, solver_attempts=1, evaluation=True, eval_manifest=manifest_path)
                with patch.dict(experiment.VALIDATION, {'cache': HeldOutBenchmark}), patch.dict(os.environ, env):
                    state = experiment.run(args, EvalModel())
            self.assertEqual(state['selection_counts']['cache'], 1)

    def test_prepare_run_serializes_eval_manifest_path(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / 'repo'; repo.mkdir(); subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            (repo / 'README').write_text('x')
            env = os.environ | {'GIT_AUTHOR_NAME': 'Test', 'GIT_AUTHOR_EMAIL': 'test@example.com',
                                'GIT_COMMITTER_NAME': 'Test', 'GIT_COMMITTER_EMAIL': 'test@example.com'}
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'initial'], check=True, env=env)
            config = {'selection_mode': 'round_robin', 'eval_manifest': Path(temp) / 'sealed.json',
                      'nested': {'paths': [Path(temp) / 'a', (Path(temp) / 'b',)]}}
            with patch.object(experiment, 'ROOT', repo), patch.dict(os.environ, env):
                state, _ = experiment.prepare_run(Path(temp) / 'run', 'path_config', 1, 'rev', None,
                                                  ['compression'], config, False)
            self.assertEqual(state['config']['eval_manifest'], str(config['eval_manifest']))
            self.assertEqual(state['config']['nested']['paths'][1], [str(Path(temp) / 'b')])
            self.assertEqual(json.loads((Path(temp) / 'run' / 'state.json').read_text())['config'], state['config'])

    def test_curriculum_actions_are_bounded_and_invalid_sample_is_not_credited(self):
        curriculum = experiment.Curriculum(['cache'])
        class Trainer:
            def sample_challenger_action(self, prompt, legal):
                self.legal = legal
                return {'action': {'benchmark': 'cache', 'difficulty': 99}}
        trainer = Trainer()
        model = SimpleNamespace(trainer=trainer)
        action, evidence = experiment.choose_curriculum_action(model, curriculum, 'cache', 'x')
        self.assertEqual(trainer.legal, curriculum.legal_actions('cache'))
        self.assertTrue(evidence['used_fallback']); self.assertFalse(evidence['action_valid'])
        self.assertEqual(action, curriculum.fallback('cache'))
        challenge = experiment.challenge_from_action(experiment.CacheBenchmark(), action, 7)
        self.assertTrue(experiment.CacheBenchmark().validate_challenge(challenge)[0])
        self.assertEqual(len(evidence['invalid_proposals']), 1)

    def test_sealed_manifest_is_stable_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'eval.json'
            judge = {'cpus': 1, 'memory_mb': 256, 'pids': 64, 'tmpfs_mb': 128,
                     'timeout_seconds': 3, 'output_bytes': 65536}
            first = experiment.create_eval_manifest(path, 42, 3, judge)
            second = experiment.load_eval_manifest(path)
            self.assertEqual(first, second)
            altered = dict(second); altered['entries'] = list(second['entries']); altered['entries'][0] = dict(altered['entries'][0])
            altered['entries'][0]['challenge'] = dict(altered['entries'][0]['challenge'])
            altered['entries'][0]['challenge']['seed'] += 1
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError): experiment.load_eval_manifest(path)
            reference_path = Path(temp) / 'reference.json'
            experiment.create_eval_manifest(reference_path, 42, 3, judge)
            with patch.object(experiment, 'stable_c_reference', return_value=('changed', {'version': 'changed'})):
                with self.assertRaises(ValueError): experiment.load_eval_manifest(reference_path)

    def test_base_final_initial_prompts_match_for_every_sealed_entry(self):
        class CaptureTrainer:
            def __init__(self): self.prompts = []
            def generate(self, role, prompt, config):
                self.prompts.append((role, prompt))
                return {'text': '{"summary":"x","changes":[{"path":"solution.c","content":"int main(void){return 0;}"}]}'}
        with tempfile.TemporaryDirectory() as temp:
            manifest = experiment.create_eval_manifest(Path(temp) / 'eval.json', 42, 3,
                {'cpus': 1, 'memory_mb': 256, 'pids': 64, 'tmpfs_mb': 128, 'timeout_seconds': 3, 'output_bytes': 65536})
            base, final = CaptureTrainer(), CaptureTrainer()
            base_model = experiment.TrainerModel(base, evaluation=True, adapter_mode='base')
            final_model = experiment.TrainerModel(final, evaluation=True, adapter_mode='trained')
            reference, _ = experiment.stable_c_reference()
            for entry in manifest['entries']:
                spec = (experiment.ROOT / 'docs' / 'benchmarks' / entry['benchmark'] / 'SPEC.md').read_text()
                for model in (base_model, final_model):
                    experiment.solve(model, spec, entry['challenge'], {}, [], '', '', entry['seeds']['solver_model'], 50, reference)
            self.assertEqual([prompt for _, prompt in base.prompts], [prompt for _, prompt in final.prompts])
            self.assertEqual([role for role, _ in base.prompts], ['base'] * 3)
            self.assertEqual([role for role, _ in final.prompts], ['solver'] * 3)

    def test_neural_repair_updates_only_verified_improvement(self):
        class RepairBenchmark(FakeBenchmark):
            def evaluate(self, files, challenge, config):
                passed = 'good' in files.get('solution.c', '')
                return {'started_at': experiment.now(), 'finished_at': experiment.now(), 'accepted': passed,
                        'build': {'exit_code': 0, 'stdout': '', 'stderr': ''},
                        'correctness': {'passed': int(passed), 'total': 1, 'cases': []},
                        'performance': {'median_ms': 100},
                        'reward_inputs': {'solver_reward': float(passed), 'challenger_reward': float(not passed)},
                        'feedback': 'pass' if passed else 'fail'}
        class NeuralTrainer:
            def __init__(self): self.updates = []; self.calls = 0
            def sample_challenger_action(self, prompt, legal): return {'action': legal[0], 'valid': True}
            def generate(self, role, prompt, config):
                self.calls += 1
                code = 'int main(void){return 1;} /* bad */' if self.calls == 1 else 'int main(void){return 0;} /* good */'
                return {'text': json.dumps({'summary': 'repair', 'changes': [{'path': 'solution.c', 'content': code}]}), 'tokens': 9}
            def update_challenger(self, value): self.updates.append(('challenger', value)); return {'updated': True}
            def update_solver(self, value): self.updates.append(('solver', value)); return {'updated': True}
            def save_checkpoint(self, path): return path
            def finalize(self): return {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repo = root / 'repo'; repo.mkdir(); subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            (repo / 'README').write_text('x'); spec = repo / 'docs/benchmarks/compression/SPEC.md'; spec.parent.mkdir(parents=True); spec.write_text('Return zero.')
            env = os.environ | {'GIT_AUTHOR_NAME': 'Test', 'GIT_AUTHOR_EMAIL': 'test@example.com', 'GIT_COMMITTER_NAME': 'Test', 'GIT_COMMITTER_EMAIL': 'test@example.com'}
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True); subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'initial'], check=True, env=env)
            args = SimpleNamespace(root=root / 'runs', run_id='repair', seed=1, hours=None, episodes=1, resume=False,
                benchmarks='compression', cpus=1, selection_mode='adaptive', memory_mb=256, pids=64, tmpfs_mb=128,
                timeout_seconds=3, output_bytes=65536, soft_gb=.1, hard_gb=.2, challenger_tokens=10, solver_tokens=50,
                solver_attempts=3, evaluation=False, trainer_backend='remote')
            trainer = NeuralTrainer()
            with patch.object(experiment, 'ROOT', repo), patch.dict(experiment.TRAIN, {'compression': RepairBenchmark}), patch.dict(os.environ, env):
                experiment.run(args, experiment.TrainerModel(trainer))
            record = json.loads((args.root / 'repair' / '000001.json').read_text())
            self.assertEqual(len(record['solver']['attempts']), 2)
            self.assertEqual([kind for kind, _ in trainer.updates], ['challenger', 'solver'])
            self.assertIn('bad', trainer.updates[1][1][0]['input']['code'])
            self.assertIn('good', trainer.updates[1][1][0]['target']['code'])

    def test_production_backend_routes_generation_updates_checkpoint_and_eval_through_trainer(self):
        class RecordingTrainer(MockTrainer):
            def __init__(self):
                super().__init__(['{"challenge":{"seed":1014,"size":1},"rationale":"r"}',
                                  '{"summary":"s","changes":[{"path":"solution.c","content":"int main(void){return 0;}"}]}'])
                self.saved = []; self.loaded = []
            def save_checkpoint(self, path): self.saved.append(str(path)); return path
            def load_checkpoint(self, path): self.loaded.append(str(path)); return {"restored": True}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repo = root / 'repo'; repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True); (repo / 'README').write_text('x')
            env = os.environ | {'GIT_AUTHOR_NAME': 'Test', 'GIT_AUTHOR_EMAIL': 'test@example.com',
                                'GIT_COMMITTER_NAME': 'Test', 'GIT_COMMITTER_EMAIL': 'test@example.com'}
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True); subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'initial'], check=True, env=env)
            spec = repo / 'docs/benchmarks/compression/SPEC.md'; spec.parent.mkdir(parents=True); spec.write_text('Return zero.')
            args = SimpleNamespace(root=root / 'runs', run_id='trainer_run', seed=5, hours=None, episodes=1, resume=False,
                benchmarks='compression', cpus=1, selection_mode='round_robin', memory_mb=256, pids=64, tmpfs_mb=128,
                timeout_seconds=3, output_bytes=65536, soft_gb=.1, hard_gb=.2, challenger_tokens=100, solver_tokens=200,
                trainer_backend='remote', evaluation=False)
            trainer = RecordingTrainer()
            with patch.object(experiment, 'ROOT', repo), patch.dict(experiment.TRAIN, {'compression': FakeBenchmark}), \
                 patch.object(experiment, 'configured_trainer', return_value=trainer), patch.dict(os.environ, env):
                experiment.run(args)
            self.assertEqual([kind for kind, _ in trainer.updates], ['challenger', 'solver'])
            self.assertTrue(trainer.saved); self.assertEqual(trainer.replies, [])
            # Evaluation uses the same generation path but never calls update/checkpoint.
            trainer = RecordingTrainer(); args.run_id, args.evaluation = 'trainer_eval', True
            with patch.object(experiment, 'ROOT', repo), patch.dict(experiment.TRAIN, {'compression': FakeBenchmark}), \
                 patch.object(experiment, 'configured_trainer', return_value=trainer), patch.dict(os.environ, env):
                experiment.run(args)
            self.assertEqual(trainer.updates, []); self.assertEqual(trainer.saved, [])
    def test_run_limits_and_graceful_shutdown_controller(self):
        args = SimpleNamespace(hours=None, episodes=None)
        state = {'episode': 4, 'deadline': None}
        controller = experiment.ShutdownController()
        self.assertTrue(experiment.should_continue(state, args, controller))
        args.hours, state['deadline'] = 1, 0
        self.assertFalse(experiment.should_continue(state, args, controller))
        args.hours, args.episodes, state['deadline'], state['episode'] = None, 3, None, 3
        self.assertFalse(experiment.should_continue(state, args, controller))
        args.episodes, state['episode'] = 5, 3
        self.assertTrue(experiment.should_continue(state, args, controller))
        controller.handler(None, None)
        self.assertTrue(controller.requested)
        self.assertFalse(experiment.should_continue(state, args, controller))
        with self.assertRaises(KeyboardInterrupt): controller.handler(None, None)
        self.assertTrue(controller.forced)

    def test_git_progress_threshold_failure_and_idempotency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); state = {'run_id': 'night', 'episode': 5, 'git_push_every': 5,
                                     'last_git_progress_episode': 0, 'last_git_commit_episode': 0,
                                     'status': 'running'}
            def result(stdout=''):
                return SimpleNamespace(stdout=stdout)
            with patch.object(experiment, 'ROOT', root), patch.object(experiment.subprocess, 'run',
                 side_effect=[result(' M experiment-progress/night.json'), result(), result(), result()] ) as call:
                self.assertTrue(experiment.git_progress(state))
            self.assertEqual(state['last_git_progress_episode'], 5)
            self.assertEqual(state['last_git_commit_episode'], 5)
            self.assertTrue(state['git_upstream_set'])
            self.assertIn('experiment/night', call.call_args_list[-1].args[0])
            # Already pushed and no changed file: no duplicate commit/push.
            with patch.object(experiment, 'ROOT', root), patch.object(experiment.subprocess, 'run', return_value=result('')) as call:
                self.assertFalse(experiment.git_progress(state, final=True))
                self.assertEqual(call.call_count, 1)
            state.update(episode=6, last_git_progress_episode=0)
            with patch.object(experiment, 'ROOT', root), patch.object(experiment.subprocess, 'run',
                 side_effect=[result(''), subprocess.CalledProcessError(1, ['git', 'push'])]):
                self.assertFalse(experiment.git_progress(state))
            self.assertIn('git', state['git_last_error'])
            # No push before a scheduled interval.
            state.update(episode=1, last_git_progress_episode=0)
            with patch.object(experiment.subprocess, 'run') as call:
                self.assertFalse(experiment.git_progress(state)); call.assert_not_called()

    def test_final_progress_push_is_only_after_safe_episode_boundary(self):
        controller = experiment.ShutdownController()
        events = []
        # The production loop checks the request between episodes; this models
        # an in-flight judge asking to stop and then completing its boundary.
        controller.handler(None, None)
        self.assertTrue(controller.requested)
        events.append('judge/update complete')
        events.append('atomic checkpoint')
        self.assertFalse(experiment.should_continue({'episode': 1, 'deadline': None}, SimpleNamespace(episodes=None), controller))
        events.append('final git push')
        self.assertEqual(events, ['judge/update complete', 'atomic checkpoint', 'final git push'])

    def test_model_delta_validation_and_invalid_solver_evidence(self):
        class Toy:
            def initialize(self, seed):
                return {'seed': seed, 'size': 1}
            def validate_challenge(self, value):
                return (value.get('seed') == 7 and value.get('size') in (1, 2, 3), 'invalid')
        class Reply:
            def __init__(self, text):
                self.text = text
            def generate(self, *args, **kwargs):
                return {'text': self.text, 'tokens': 12, 'truncated': False}
        challenge, rationale, invalid, _ = experiment.choose_challenge(
            Reply('{"parameters":{"size":2},"rationale":"probe range"}'), Toy(), 'Public spec',
            7, [], [], 50)
        self.assertEqual(challenge, {'seed': 7, 'size': 2})
        self.assertEqual(rationale, 'probe range')
        self.assertFalse(invalid)
        source, changed, _, raw, _, generation = experiment.solve(
            Reply('unstructured failure'), 'Public spec', challenge, {}, [], '', '', 9, 50)
        self.assertEqual((source, changed, raw), ({}, [], 'unstructured failure'))
        self.assertIn('error', generation)

    def test_safe_paths_and_bounded_memory(self):
        with self.assertRaises(ValueError):
            apply_changes({'solution.c': ''}, {'changes': [{'path': '../escape.c', 'content': ''}]})
        memory = []
        for i in range(8):
            memory = experiment.push_memory(memory, str(i))
        self.assertEqual(memory, ['3', '4', '5', '6', '7'])
        self.assertEqual(parse_solver_response('```c\nint main(void){return 0;}\n```')['changes'][0]['path'],
                         'solution.c')
        self.assertEqual(read_range({'solution.c': 'a\nb\nc\n'}, 'solution.c', 2, 1), 'b\n')
        self.assertEqual(lexical_search({'solution.c': 'a\nneedle\n'}, {'needle'})[0][2], 2)

    def test_acceptance_noise_floor(self):
        self.assertEqual(experiment.accept({'accepted': True, 'performance': {'median_ms': 94}},
                                           {'accepted': True, 'performance': {'median_ms': 100}})[0], False)
        self.assertEqual(experiment.accept({'accepted': True, 'performance': {'median_ms': 80}},
                                           {'accepted': True, 'performance': {'median_ms': 100}})[0], True)

    def test_adaptive_selection_rng_survives_json_checkpoint(self):
        state = {'selection_mode': 'adaptive', 'benchmarks': ['compression', 'csv'],
                 'episode': 0, 'Q_challenger': experiment.init_q(['compression', 'csv']),
                 'performance_history': [], 'policy_rng_state': random.Random(42).getstate(),
                 'epsilon': .3}
        first = experiment.select_benchmark(state)
        state['policy_rng_state'] = first[2]
        restored = json.loads(json.dumps(state))
        self.assertEqual(experiment.select_benchmark(state), experiment.select_benchmark(restored))

    def test_resume_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / 'repo'
            repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            (repo / 'README').write_text('test')
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            env = os.environ | {'GIT_AUTHOR_NAME': 'Test', 'GIT_AUTHOR_EMAIL': 'test@example.com',
                                'GIT_COMMITTER_NAME': 'Test', 'GIT_COMMITTER_EMAIL': 'test@example.com'}
            subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'initial'], check=True, env=env)
            spec = repo / 'docs/benchmarks/compression/SPEC.md'
            spec.parent.mkdir(parents=True)
            spec.write_text('Return zero. Specify candidate code.')
            args = SimpleNamespace(root=root / 'runs', run_id='test_run', seed=5, hours=None,
                                   episodes=1, resume=False, benchmarks='compression', cpus=1,
                                   selection_mode='round_robin',
                                   memory_mb=256, pids=64, tmpfs_mb=128, timeout_seconds=3,
                                   output_bytes=65536, soft_gb=.1, hard_gb=.2,
                                   challenger_tokens=100, solver_tokens=200)
            with patch.object(experiment, 'ROOT', repo), patch.dict(experiment.TRAIN, {'compression': FakeBenchmark}), patch.dict(os.environ, env):
                first = experiment.run(args, FakeModel())
                self.assertEqual(first['episode'], 1)
                self.assertEqual(len(first['solver_memory']), 1)
                record = json.loads((args.root / args.run_id / '000001.json').read_text())
                self.assertEqual(record['outcome'], 'accepted')
                self.assertTrue(record['git_after'])
                self.assertEqual(len(record['challenger']['memory_after']), 1)
                args.resume, args.episodes = True, 2
                second = experiment.run(args, FakeModel())
                self.assertEqual(second['episode'], 2)
                self.assertEqual(len(second['solver_memory']), 2)
                self.assertEqual(json.loads((args.root / args.run_id / '000002.json').read_text())['episode_id'], 2)
                self.assertTrue((args.root / args.run_id / 'report.md').exists())
                self.assertFalse((args.root / args.run_id / 'pending.json').exists())
                run_dir = args.root / args.run_id
                workspace = run_dir / 'workspace'
                # A crashed, unjudged response is retained and never accepted.
                experiment.atomic_json(run_dir / 'pending.json',
                                       {'ready': False, 'episode_id': 3, 'candidate': {'solution.c': 'bad'}})
                unchanged = experiment.reconcile(run_dir, second, workspace)
                self.assertEqual(unchanged['accepted_head'], second['accepted_head'])
                self.assertEqual(unchanged['episode'], 2)
                self.assertEqual(len(list(run_dir.glob('interrupted-000003-*.json'))), 1)
                # A commit made just before a crash is linked to the pending evidence on resume.
                code = 'int main(void){return 0;} /* recovered */'
                experiment.install_files(workspace, 'compression', {'solution.c': code})
                subprocess.run(['git', '-C', str(workspace), 'add', '--', 'solutions/compression'], check=True)
                message = 'feat(experiment): accept compression episode 3'
                subprocess.run(['git', '-C', str(workspace), 'commit', '-qm', message], check=True, env=env)
                recovered_record = dict(record)
                recovered_record.update({'episode_id': 3, 'git_before': second['accepted_head'], 'git_after': None})
                next_state = dict(second) | {'episode': 3}
                experiment.atomic_json(run_dir / 'pending.json',
                                       {'ready': True, 'episode_id': 3, 'record': recovered_record,
                                        'files': {'solution.c': code}, 'next_state': next_state,
                                        'commit_message': message})
                recovered = experiment.reconcile(run_dir, second, workspace)
                self.assertEqual(recovered['episode'], 3)
                self.assertEqual(json.loads((run_dir / '000003.json').read_text())['git_after'],
                                 recovered['accepted_head'])
                self.assertGreater(experiment.run_bytes(run_dir, recovered, workspace),
                                   experiment.storage_bytes(run_dir))
                args.run_id, args.resume, args.episodes, args.selection_mode = 'adaptive_run', False, 1, 'adaptive'
                adaptive = experiment.run(args, FakeModel())
                self.assertEqual(adaptive['episode'], 1)
                adaptive_record = json.loads((args.root / args.run_id / '000001.json').read_text())
                self.assertEqual(adaptive_record['selection_policy']['mode'], 'adaptive')
                self.assertIn('Q_challenger', adaptive)


if __name__ == '__main__':
    unittest.main()
