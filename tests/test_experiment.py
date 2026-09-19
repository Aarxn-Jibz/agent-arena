import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arena import experiment
from arena.experiment_context import apply_changes
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
    def test_safe_paths_and_bounded_memory(self):
        with self.assertRaises(ValueError):
            apply_changes({'solution.c': ''}, {'changes': [{'path': '../escape.c', 'content': ''}]})
        memory = []
        for i in range(8):
            memory = experiment.push_memory(memory, str(i))
        self.assertEqual(memory, ['3', '4', '5', '6', '7'])

    def test_acceptance_noise_floor(self):
        self.assertEqual(experiment.accept({'accepted': True, 'performance': {'median_ms': 94}},
                                           {'accepted': True, 'performance': {'median_ms': 100}})[0], False)
        self.assertEqual(experiment.accept({'accepted': True, 'performance': {'median_ms': 80}},
                                           {'accepted': True, 'performance': {'median_ms': 100}})[0], True)

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


if __name__ == '__main__':
    unittest.main()
