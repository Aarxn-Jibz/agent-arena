"""Offline Challenger/Solver experiment. Only benchmark judges execute candidate C."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt

from .compression import CompressionBenchmark
from .csv_benchmark import CsvBenchmark
from .expression import ExpressionBenchmark
from .graph_benchmark import GraphBenchmark
from .http_benchmark import HttpBenchmark
from .secondary import CacheBenchmark, LogBenchmark, SearchBenchmark
from .evidence import viewer_events, write_episode
from .experiment_context import parse_object, selected_context, source_diff
from .sandbox import SandboxConfig
from .marl import choose_action, init_q, q_update, rolling_average, skill_state, challenger_reward as frontier_reward
from .training import (PRODUCTION_MODEL, BenchmarkProgress, Curriculum, HFPEFTTrainer, MockTrainer,
                       ModelConfig, RemoteTrainer, RemoteTrainerConfig, SolverResponse, failure_feedback,
                       parse_solver_contract, reference_identity, verified_correction, challenger_reward as curriculum_reward)

TRAIN = {'compression': CompressionBenchmark, 'csv': CsvBenchmark, 'http': HttpBenchmark,
         'expression': ExpressionBenchmark, 'graph': GraphBenchmark}
VALIDATION = {'cache': CacheBenchmark, 'log': LogBenchmark, 'search': SearchBenchmark}
ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = ROOT / 'references'


class TrainerModel:
    """Small compatibility shim: the existing arena prompt construction speaks Trainer."""
    def __init__(self, trainer, revision: str | None = None, *, evaluation: bool = False, adapter_mode: str = "trained"):
        self.trainer = trainer
        self.revision = revision or getattr(getattr(trainer, "model_config", None), "revision", None) or "configured"
        self.evaluation, self.adapter_mode = evaluation, adapter_mode
        self.prompts = {}
    def generate(self, messages, *, seed, max_new_tokens):
        role = "challenger" if "Challenger" in messages[0]["content"] else "solver"
        if self.evaluation and self.adapter_mode == "base": role = "base"
        prompt = "\n\n".join(message["content"] for message in messages)
        self.prompts[role] = prompt
        value = self.trainer.generate(role, prompt, {"max_new_tokens": max_new_tokens, "seed": seed})
        if isinstance(value, dict):
            return {"text": str(value.get("text", value.get("output", ""))), "tokens": value.get("tokens", 0),
                    "truncated": bool(value.get("truncated", False)), "input_prompt": prompt,
                    "raw_generation": value.get("raw_generation", value.get("text", value.get("output", ""))), **value}
        return {"text": str(value), "tokens": 0, "truncated": False}


def configured_trainer(args):
    backend = getattr(args, 'trainer_backend', 'legacy')
    if backend == 'mock': return MockTrainer()
    if backend == 'legacy': return None
    if backend == 'hf': return HFPEFTTrainer(ModelConfig(revision=getattr(args, 'model_revision', None)), allow_download=False)
    token = os.environ.get(getattr(args, 'remote_token_env', 'ARENA_REMOTE_TOKEN'), '')
    return RemoteTrainer(RemoteTrainerConfig(url=args.trainer_url, token=token, run_id=args.run_id,
                         revision=getattr(args, 'model_revision', None), timeout=args.remote_timeout,
                         retries=args.remote_retries, hf_repo=getattr(args, 'hf_repo', None),
                         hf_push_every=getattr(args, 'hf_push_every', 5),
                         hf_resume_push_every=getattr(args, 'hf_resume_push_every', 10)))


class ShutdownController:
    """First signal stops at an episode boundary; a second one aborts now."""
    def __init__(self): self.requested = self.forced = False; self._old = {}
    def handler(self, signum, frame):
        if self.requested:
            self.forced = True
            raise KeyboardInterrupt
        self.requested = True
        print('Shutdown requested; finishing current episode and checkpointing...', flush=True)
    def install(self):
        for item in (signal.SIGINT, getattr(signal, 'SIGTERM', None)):
            if item is not None: self._old[item] = signal.signal(item, self.handler)
    def restore(self):
        for item, previous in self._old.items(): signal.signal(item, previous)


def should_continue(state: dict, args, shutdown: ShutdownController | None = None) -> bool:
    if shutdown and shutdown.requested: return False
    if args.episodes is not None and state['episode'] >= args.episodes: return False
    return state.get('deadline') is None or time.time() < state['deadline']


def progress_path(run_id: str, workspace: Path | None = None) -> Path:
    workspace = ROOT if workspace is None else workspace
    return workspace / 'experiment-progress' / f'{run_id}.json'


def write_progress(state: dict, workspace: Path | None = None):
    """Only compact state goes into Git; evidence/checkpoints remain local."""
    value = {key: state.get(key) for key in ('run_id', 'episode', 'started_at', 'benchmarks', 'model_id',
             'model_revision', 'selection_counts', 'performance_history', 'git_push_every',
             'last_git_progress_episode', 'last_git_commit_episode', 'status')}
    workspace = ROOT if workspace is None else workspace
    path = progress_path(state['run_id'], workspace); atomic_json(path, value); return path


def git_progress(state: dict, workspace: Path | None = None, *, final: bool = False) -> bool:
    """Best-effort progress visibility. Never raises into the training loop."""
    workspace = ROOT if workspace is None else workspace
    every = int(state.get('git_push_every', 0))
    if not every: return False
    episode, pushed = state['episode'], int(state.get('last_git_progress_episode', 0))
    if not final and episode - pushed < every: return False
    path = write_progress(state, workspace)
    try:
        relative = str(path.relative_to(workspace))
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', relative], cwd=workspace,
                               text=True, capture_output=True, check=True).stdout.strip()
        committed = int(state.get('last_git_commit_episode', 0))
        if dirty:
            subprocess.run(['git', 'add', '--', relative], cwd=workspace, check=True)
            start = committed + 1
            subprocess.run(['git', 'commit', '-m', f'experiment({state["run_id"]}): episodes {start}-{episode}'],
                           cwd=workspace, check=True)
            state['last_git_commit_episode'] = episode
        if state.get('last_git_progress_episode', 0) < episode:
            command = (['git', 'push'] if state.get('git_upstream_set')
                       else ['git', 'push', '--set-upstream', 'origin', state.get('branch', 'experiment/' + state['run_id'])])
            subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True)
            state['last_git_progress_episode'] = episode
            state['git_upstream_set'] = True
        state['git_last_error'] = None
        return bool(dirty)
    except (OSError, subprocess.CalledProcessError) as err:
        state['git_last_error'] = str(err)
        print(f'Git progress push failed (continuing): {err}', flush=True)
        return False


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def json_safe(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    return value


def git(*args, cwd=None):
    p = subprocess.run(['git', *args], cwd=cwd or ROOT, text=True, capture_output=True, check=True)
    return p.stdout.strip()


def memory_note(text, limit=600):
    return ' '.join(str(text).split())[:limit]


def push_memory(items: list[str], note: str, limit=600):
    return (items + [memory_note(note, limit)])[-5:]


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


def select_benchmark(state: dict):
    if state['selection_mode'] == 'round_robin':
        return state['benchmarks'][state['episode'] % len(state['benchmarks'])], None, None
    rng = random.Random()
    rng.setstate(_tuple_tree(state['policy_rng_state']))
    performance_state = skill_state(rolling_average(state['performance_history']))
    name = choose_action(state['Q_challenger'][performance_state], state['epsilon'], rng)
    return name, performance_state, rng.getstate()


def curriculum_from_state(state: dict) -> Curriculum:
    curriculum = Curriculum(state['benchmarks'])
    saved = state.get('curriculum')
    if saved:
        curriculum.progress = {name: BenchmarkProgress(**saved['progress'][name]) for name in state['benchmarks']}
        curriculum.by_level = {name: {int(level): trials for level, trials in levels.items()}
                               for name, levels in saved['by_level'].items()}
    return curriculum


def curriculum_state(curriculum: Curriculum) -> dict:
    return {'progress': {name: asdict(value) for name, value in curriculum.progress.items()},
            'by_level': curriculum.by_level}


def challenge_from_action(benchmark, action: dict, seed: int) -> dict:
    """Deterministic, mechanically-valid curriculum action translation."""
    challenge = benchmark.initialize(seed)
    level = action['difficulty']
    # The validation tasks are held out, but levels still select bounded workload knobs.
    if getattr(benchmark, 'name', '') == 'cache':
        challenge.update(capacity=min(128, 4 * level), operations=min(1000, 30 * level),
                         keys=min(256, 8 * level), distribution=('uniform', 'hot', 'sequential')[(level - 1) % 3])
    elif getattr(benchmark, 'name', '') == 'log':
        challenge.update(records=min(10000, 100 * level), malformed=min(100, level - 1),
                         distribution=('uniform', 'bursty')[(level - 1) % 2],
                         query=('count', 'by_level', 'filter')[(level - 1) % 3])
    elif getattr(benchmark, 'name', '') == 'search':
        challenge.update(documents=min(1000, 20 * level), queries=min(30, 4 + level),
                         words_per_doc=min(100, 10 * level),
                         distribution=('uniform', 'skewed')[(level - 1) % 2], matching=('exact', 'prefix')[(level - 1) % 2])
    valid, reason = benchmark.validate_challenge(challenge)
    if not valid: raise ValueError(f'curriculum produced invalid challenge: {reason}')
    return challenge


def stable_c_reference() -> tuple[str, dict]:
    text = (REFERENCE_ROOT / 'c-library.md').read_text()
    return text, reference_identity(text, 'references/c-library.md')


def create_eval_manifest(path: Path, seed: int, solver_attempts: int, judge: dict) -> dict:
    """Seal the held-out cache/log/search inputs before either comparison."""
    entries = []
    reference, reference_meta = stable_c_reference()
    for index, name in enumerate(('cache', 'log', 'search'), 1):
        challenge_seed = (seed + index * 1009) % (2**32)
        benchmark = VALIDATION[name]()
        challenge = benchmark.initialize(challenge_seed)
        valid, reason = benchmark.validate_challenge(challenge)
        if not valid: raise ValueError(f'invalid evaluation challenge: {name}: {reason}')
        spec = (ROOT / 'docs' / 'benchmarks' / name / 'SPEC.md').read_text()
        entries.append({'benchmark': name, 'challenge': challenge,
                        'seeds': {'challenge': challenge_seed, 'solver_model': challenge_seed + 500000},
                        'reference': reference_meta,
                        'spec_sha256': hashlib.sha256(spec.encode()).hexdigest()})
    manifest = {'version': 1, 'entries': entries, 'solver_attempts': solver_attempts,
                'protocol': 'experiment-solver-json-v1', 'judge': judge}
    manifest['sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    atomic_json(path, manifest)
    return manifest


def load_eval_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    claimed = manifest.pop('sha256', None)
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest['sha256'] = claimed
    if claimed != actual or not isinstance(manifest.get('entries'), list) or not manifest['entries']:
        raise ValueError('invalid sealed evaluation manifest')
    if any(entry.get('benchmark') not in VALIDATION for entry in manifest['entries']):
        raise ValueError('sealed evaluation manifest contains an unknown benchmark')
    _, reference = stable_c_reference()
    if any(entry.get('reference') != reference for entry in manifest['entries']):
        raise ValueError('sealed evaluation C reference differs from references/c-library.md')
    return manifest


def choose_curriculum_action(model: TrainerModel, curriculum: Curriculum, benchmark: str, prompt: str):
    legal = curriculum.legal_actions(benchmark)
    sampled = model.trainer.sample_challenger_action(prompt, legal)
    action = sampled.get('action') if isinstance(sampled, dict) else None
    valid, reason = curriculum.validate(action)
    if action not in legal:
        valid, reason = False, 'action was not one of the supplied legal actions'
    used_fallback = not valid
    if used_fallback: action = curriculum.fallback(benchmark)
    evidence = {'legal_actions': legal, 'sampled': sampled, 'action_valid': valid,
                'used_fallback': used_fallback, 'invalid_proposals': ([] if valid else [{'proposal': sampled, 'error': reason}])}
    return action, evidence


def project_files(workspace: Path, benchmark: str):
    root = workspace / 'solutions' / benchmark
    return {str(path.relative_to(root)): path.read_text() for path in sorted(root.rglob('*'))
            if path.is_file() and path.suffix in ('.c', '.h')}


def install_files(workspace: Path, benchmark: str, files: dict[str, str]):
    root = workspace / 'solutions' / benchmark
    root.mkdir(parents=True, exist_ok=True)
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.c', '.h') and str(path.relative_to(root)) not in files:
            path.unlink()
    for name, content in files.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def storage_bytes(path: Path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.exists() else 0


def run_bytes(run_dir: Path, state: dict, workspace: Path):
    total = storage_bytes(run_dir)
    base = state.get('base_head', state['accepted_head'])
    objects = git('rev-list', '--objects', 'HEAD', '^' + base, cwd=workspace)
    if objects:
        ids = [line.split()[0] for line in objects.splitlines()]
        process = subprocess.run(['git', 'cat-file', '--batch-check=%(objectsize)'],
                                 cwd=workspace, input='\n'.join(ids) + '\n',
                                 text=True, capture_output=True, check=True)
        total += sum(int(line) for line in process.stdout.splitlines())
    return total


def safe_report(run_dir: Path, state: dict):
    lines = [f'# Experiment {state["run_id"]}', '', f'- Episodes: {state["episode"]}',
             f'- Started: {state["started_at"]}', f'- Deadline: {state.get("deadline")}',
             f'- Seed: {state["seed"]}', f'- Benchmarks: {", ".join(state["benchmarks"])}',
             f'- Selection: {state.get("selection_mode", "round_robin")}; counts {state["selection_counts"]}',
             f'- Token budgets: Challenger {state["config"]["challenger_tokens"]}, Solver {state["config"]["solver_tokens"]}',
             f'- Run quota: soft {state["config"]["soft_gb"]} GiB, hard {state["config"]["hard_gb"]} GiB',
             f'- Accepted Git HEAD: {state["accepted_head"]}',
             f'- Model: {state["model_id"]} @ {state["model_revision"]}',
             f'- Frozen model weights: yes', '', '## Outcomes', '']
    records = []
    for path in sorted(run_dir.glob('[0-9]*.json')):
        try:
            item = json.loads(path.read_text())
            records.append(item)
            lines.append(f'- {item["episode_id"]} {item["benchmark"]}: {item["outcome"]}; '
                         f'{item["judge"]["feedback"]}; Git {item["git_after"] or "unchanged"}')
        except (ValueError, KeyError):
            pass
    if records:
        lines += ['', '## Aggregate', '',
                  f'- Accepted: {sum(x["outcome"] == "accepted" for x in records)}/{len(records)}',
                  f'- Average Solver reward: {sum(x["rewards"]["solver"] for x in records)/len(records):.3f}',
                  f'- Average Challenger reward: {sum(x["rewards"]["challenger"] for x in records)/len(records):.3f}',
                  '- These are local Judge outcomes, not a held-out improvement claim.']
    if state.get('selection_mode') == 'adaptive':
        lines += ['', '## Learned benchmark policy', '',
                  'The Q table estimates the usefulness of benchmark selection by recent pass-rate state.',
                  'It is a surrounding selection policy; SmolLM2 weights are frozen.', '']
        for level, row in state['Q_challenger'].items():
            preferred = max(row, key=row.get)
            lines.append(f'- {level}: {preferred} (Q={row[preferred]:.3f})')
    path = run_dir / 'report.md'
    tmp = path.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    os.replace(tmp, path)


def rebuild_events(run_dir: Path):
    lines = []
    for path in sorted(run_dir.glob('[0-9]*.json')):
        try:
            record = json.loads(path.read_text())
            lines.extend(json.dumps(event, ensure_ascii=False) for event in viewer_events(record))
        except (ValueError, KeyError):
            continue
    target = run_dir / 'events.jsonl'
    temp = target.with_suffix('.tmp')
    temp.write_text('\n'.join(lines) + ('\n' if lines else ''))
    os.replace(temp, target)


def primary_ms(evaluation: dict):
    p = evaluation['performance']
    for key in ('median_ms', 'compression_median_ms', 'p95_latency_ms', 'median_latency_ms'):
        value = p.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    return None


def accept(candidate: dict, baseline: dict | None, benchmark: str = '') -> tuple[bool, str]:
    if not candidate['accepted']:
        return False, 'Judge rejected correctness or resource limits'
    if baseline is None or not baseline['accepted']:
        return True, 'first passing candidate or correctness improvement'
    if benchmark == 'compression':
        old_cases = baseline['performance'].get('cases', [])
        new_cases = candidate['performance'].get('cases', [])
        if old_cases and len(old_cases) == len(new_cases):
            original = sum(x['original_bytes'] for x in old_cases)
            gain = sum(x['compressed_bytes'] for x in old_cases) - sum(x['compressed_bytes'] for x in new_cases)
            old_unpack = baseline['performance'].get('decompression_median_ms')
            new_unpack = candidate['performance'].get('decompression_median_ms')
            old_pack = baseline['performance'].get('compression_median_ms')
            new_pack = candidate['performance'].get('compression_median_ms')
            if (gain >= max(16, original * .05) and
                    all(a is None or b is None or b <= a * 1.25 + 5
                        for a, b in ((old_pack, new_pack), (old_unpack, new_unpack)))):
                return True, 'at least 5% and 16 bytes smaller without material speed regression'
    old, new = primary_ms(baseline), primary_ms(candidate)
    if old is None or new is None:
        return False, 'no comparable performance metric'
    if old - new >= max(5.0, old * 0.10):
        if benchmark == 'compression':
            old_size = sum(x['compressed_bytes'] for x in baseline['performance'].get('cases', []))
            new_size = sum(x['compressed_bytes'] for x in candidate['performance'].get('cases', []))
            if new_size > old_size + max(16, old_size * .05):
                return False, 'speed gain costs more than 5% and 16 compressed bytes'
        return True, 'at least 10% and 5 ms faster with full correctness'
    return False, 'performance difference below 10% and 5 ms noise threshold'


def synthetic_failure(benchmark, challenge, reason):
    stamp = now()
    return {'started_at': stamp, 'finished_at': stamp,
            'build': {'exit_code': None, 'stdout': '', 'stderr': reason[:65536]},
            'correctness': {'passed': 0, 'total': 1, 'cases': []}, 'performance': {},
            'reward_inputs': {'solver_reward': 0.0, 'challenger_reward': 1.0},
            'feedback': reason[:1000], 'accepted': False}


def judge_candidate(benchmark, files: dict[str, str], challenge: dict, config: SandboxConfig):
    """Training benchmarks consume files; held-out C benchmarks consume solution.c."""
    source = files.get('solution.c', '') if getattr(benchmark, 'name', '') in VALIDATION else files
    return benchmark.evaluate(source, challenge, config)


def prepare_run(run_dir: Path, run_id: str, seed: int, model_revision: str, hours: float | None,
                benchmarks: list[str], config: dict, resume: bool):
    config = json_safe(config)
    state_path = run_dir / 'state.json'
    workspace = run_dir / 'workspace'
    if resume:
        state = json.loads(state_path.read_text())
        state.setdefault('selection_mode', 'round_robin')
        state.setdefault('Q_challenger', init_q(benchmarks))
        state.setdefault('performance_history', [])
        state.setdefault('policy_rng_state', random.Random(seed).getstate())
        state.setdefault('alpha', .4)
        state.setdefault('gamma', .8)
        state.setdefault('epsilon', .3)
        state.setdefault('git_push_every', 0)
        state.setdefault('last_git_progress_episode', 0)
        state.setdefault('last_git_commit_episode', 0)
        state.setdefault('git_upstream_set', False)
        state.setdefault('status', 'running')
        state['config'].setdefault('git_push_every', 0)
        state['config'].setdefault('selection_mode', 'round_robin')
        if (state['run_id'] != run_id or state['seed'] != seed or state['benchmarks'] != benchmarks
                or state['model_revision'] != model_revision):
            raise ValueError('resume configuration differs from checkpoint')
        for key in ('selection_mode', 'challenger_tokens', 'solver_tokens', 'cpus', 'memory_mb', 'pids',
                    'tmpfs_mb', 'timeout_seconds', 'output_bytes', 'soft_gb', 'hard_gb', 'git_push_every'):
            if state['config'][key] != config.get(key, 0):
                raise ValueError(f'resume {key} differs from checkpoint')
        return state, workspace
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('run directory already exists; use --resume')
    run_dir.mkdir(parents=True, exist_ok=True)
    branch = 'experiment/' + run_id
    git('worktree', 'add', '-b', branch, str(workspace), 'HEAD')
    head = git('rev-parse', 'HEAD', cwd=workspace)
    started = time.time()
    policy_rng = random.Random(seed)
    state = {'run_id': run_id, 'seed': seed, 'episode': 0, 'started_at': now(),
             'deadline': started + hours * 3600 if hours else None,
             'benchmarks': benchmarks, 'model_id': PRODUCTION_MODEL,
             'model_revision': model_revision, 'config': config, 'branch': branch,
             'accepted_head': head, 'base_head': head,
             'challenger_memory': [], 'solver_memory': [],
             'selection_counts': {name: 0 for name in benchmarks},
             'selection_mode': config['selection_mode'], 'Q_challenger': init_q(benchmarks),
             'performance_history': [], 'policy_rng_state': policy_rng.getstate(),
             'alpha': .4, 'gamma': .8, 'epsilon': .3}
    state.update(git_push_every=config.get('git_push_every', 0), last_git_progress_episode=0,
                 last_git_commit_episode=0, git_upstream_set=False, status='running')
    atomic_json(state_path, state)
    atomic_json(run_dir / 'manifest.json', state)
    return state, workspace


def reconcile(run_dir: Path, state: dict, workspace: Path):
    pending = run_dir / 'pending.json'
    if not pending.exists():
        if git('rev-parse', 'HEAD', cwd=workspace) != state['accepted_head']:
            raise RuntimeError('accepted worktree changed outside experiment')
        rebuild_events(run_dir)
        return state
    transaction = json.loads(pending.read_text())
    episode = transaction['episode_id']
    head = git('rev-parse', 'HEAD', cwd=workspace)
    if transaction.get('ready'):
        record = transaction['record']
        if record['outcome'] == 'accepted':
            if head == state['accepted_head']:
                install_files(workspace, record['benchmark'], transaction['files'])
                git('add', '--', f'solutions/{record["benchmark"]}', cwd=workspace)
                git('commit', '-m', transaction['commit_message'], cwd=workspace)
                head = git('rev-parse', 'HEAD', cwd=workspace)
            elif (git('rev-parse', 'HEAD^', cwd=workspace) != state['accepted_head'] or
                  git('log', '-1', '--format=%B', cwd=workspace) != transaction['commit_message']):
                raise RuntimeError('unexpected accepted commit during recovery')
            record['git_after'] = head
        elif head != state['accepted_head']:
            raise RuntimeError('unexpected Git HEAD during rejected episode')
        if not (run_dir / f'{episode:06d}.json').exists():
            write_episode(run_dir.parent, record)
        state = transaction['next_state']
        state['accepted_head'] = head
        atomic_json(run_dir / 'state.json', state)
    else:
        if head != state['accepted_head']:
            raise RuntimeError('incomplete unjudged episode changed Git HEAD')
        interrupted = run_dir / f'interrupted-{episode:06d}-{int(time.time())}.json'
        os.replace(pending, interrupted)
        rebuild_events(run_dir)
        return state
    pending.unlink()
    rebuild_events(run_dir)
    safe_report(run_dir, state)
    return state


def choose_challenge(model, benchmark, public_spec: str, seed: int, memory: list[str],
                     recent: list[dict], max_tokens: int):
    schema = benchmark.initialize(seed)
    example_key = next(key for key in schema if key != 'seed')
    system = ('You are Challenger. Choose valid test parameters, never implementation ideas. '
              'Output a short JSON object only.')
    prompt = (f'Public specification:\n{public_spec[:3000]}\n'
              f'Recent objective Judge outcomes: {json.dumps(recent[-3:])[:800]}\n'
              f'Your last five interpretations (may be wrong): {json.dumps(memory[-5:])[:800]}\n'
              f'Default parameters: {json.dumps(schema)}. The seed is fixed to {seed}.\n'
              'Choose one or more non-seed fields to vary within the public bounds. '
              f'Return ONLY JSON like {json.dumps({"parameters": {example_key: schema[example_key]}, "rationale": "probe a weakness"})}. '
              'Include only fields you change. No Markdown.')
    attempts = []
    for retry in range(2):
        generated = model.generate([{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': prompt}],
                                   seed=seed + retry, max_new_tokens=max_tokens)
        try:
            data = parse_object(generated['text'])
            if isinstance(data.get('parameters'), dict):
                challenge = schema | data['parameters']
            else:
                challenge = data.get('challenge', data)
            valid, reason = benchmark.validate_challenge(challenge)
            if valid and challenge.get('seed') == seed:
                return challenge, memory_note(data.get('rationale', ''), 500), attempts, generated
            attempts.append({'response': generated['text'][:2000], 'error': reason})
        except (ValueError, KeyError, TypeError) as err:
            attempts.append({'response': generated['text'][:2000], 'error': str(err)})
    return schema, 'Validated fallback after two invalid Challenger proposals', attempts, generated


def solve(model, public_spec: str, challenge: dict, files: dict[str, str], memory: list[str],
          latest_diff: str, failures: str, seed: int, max_tokens: int, reference: str = ''):
    context, reads = selected_context(files, challenge, latest_diff, failures, ROOT / 'references')
    system = ('You are a C programmer. Produce candidate source for TinyCC. '
              'No external libraries or network. Output only the required tagged contract.')
    user = (f'Public specification:\n{public_spec[:5500]}\n'
            f'Accepted source context:\n{context}\n'
            f'Previous Judge feedback: {failures[:800]}\n'
            f'C reference:\n{reference[:6000]}\n'
            f'Your last five notes (may be wrong): {json.dumps(memory[-5:])[:1000]}\n\n'
            f'Now solve this validated challenge: {json.dumps(challenge)}\n'
            'Begin with <CODE>. Output exactly one complete C program. '
            'End with </CODE>. Output nothing else.')
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    generated = model.generate(messages, seed=seed, max_new_tokens=max_tokens)
    text = generated['text']
    parsed = parse_solver_contract(text)
    if parsed.malformed:
        generated['error'] = parsed.malformed
        return files, [], '', text, reads, generated
    updated = dict(files); updated['solution.c'] = parsed.code
    changed = ['solution.c'] if files.get('solution.c') != parsed.code else []
    return updated, changed, memory_note(parsed.strategy, 500), text, reads, generated


def episode(model, state: dict, workspace: Path, run_dir: Path, config: SandboxConfig,
            challenger_tokens: int, solver_tokens: int, solver_attempts: int = 1):
    number = state['episode'] + 1
    neural = isinstance(model, TrainerModel) and hasattr(model.trainer, 'sample_challenger_action')
    eval_entry = state.get('eval_manifest_entries', [])[state['episode']] if state.get('eval_manifest_entries') else None
    # Neural Challenger selects difficulty, not an arbitrary benchmark parameter object.
    name, performance_state, next_rng_state = ((eval_entry['benchmark'], None, None) if eval_entry else
        ((state['benchmarks'][state['episode'] % len(state['benchmarks'])], None, None) if neural else select_benchmark(state)))
    benchmark = TRAIN.get(name, VALIDATION.get(name))()
    seed = (state['seed'] + number * 1009) % (2**32)
    spec = (ROOT / 'docs/benchmarks' / name / 'SPEC.md').read_text()
    recent = []
    for path in sorted(run_dir.glob('[0-9]*.json'))[-3:]:
        item = json.loads(path.read_text())
        recent.append({'benchmark': item['benchmark'], 'feedback': item['judge']['feedback'],
                       'outcome': item['outcome'], 'correctness': item['correctness']['passed'],
                       'total': item['correctness']['total'],
                       'performance': primary_ms(item)})
    curriculum = curriculum_from_state(state) if neural else None
    if eval_entry:
        action, challenger_evidence, challenger_gen = None, {'action_valid': False, 'used_fallback': True, 'invalid_proposals': []}, None
        challenge, rationale, invalid = eval_entry['challenge'], 'Sealed evaluation manifest', []
        seed = eval_entry['seeds']['challenge']
        solver_attempts = state.get('eval_solver_attempts', solver_attempts)
    elif neural:
        action_prompt = f'Choose one legal curriculum action only. Benchmark: {name}. Recent judge outcomes: {json.dumps(recent[-3:])}'
        action, challenger_evidence = choose_curriculum_action(model, curriculum, name, action_prompt)
        challenge, rationale = challenge_from_action(benchmark, action, seed), f'Curriculum action {action}'
        invalid, challenger_gen = challenger_evidence['invalid_proposals'], challenger_evidence['sampled']
    else:
        action, challenger_evidence = None, None
        challenge, rationale, invalid, challenger_gen = choose_challenge(
            model, benchmark, spec, seed, state['challenger_memory'], recent, challenger_tokens)
    atomic_json(run_dir / 'pending.json', {'ready': False, 'episode_id': number,
                                           'challenge': challenge, 'rationale': rationale,
                                           'challenger_generation': challenger_gen})
    before = (dict(state['eval_initial_files'][name]) if eval_entry else project_files(workspace, name))
    latest_diff = '' if eval_entry else git('show', '--format=', '--', f'solutions/{name}', cwd=workspace)[:1500]
    failures = '' if eval_entry else '\n'.join(x['feedback'] for x in recent if x['outcome'] == 'rejected')
    attempts, attempt_evidence = [], []
    reference_text, reference = stable_c_reference()
    if eval_entry and eval_entry['reference'] != reference:
        raise ValueError('sealed evaluation C reference changed during run')
    candidate, changed, summary, response, reads, solver_gen = before, [], '', '', [], {}
    result = synthetic_failure(benchmark, challenge, 'Solver made no attempt')
    feedback = failures
    for attempt_number in range(1, max(1, solver_attempts) + 1):
        try:
            candidate, changed, summary, response, reads, solver_gen = solve(
                model, spec, challenge, before, [] if eval_entry else state['solver_memory'], latest_diff, feedback,
                seed + 500000 + attempt_number - 1, solver_tokens, reference_text)
            result = (synthetic_failure(benchmark, challenge, 'Invalid Solver response: ' + solver_gen['error'])
                      if solver_gen.get('error') else judge_candidate(benchmark, candidate, challenge, config))
        except (ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as err:
            candidate, changed, summary, response, reads, solver_gen = before, [], '', str(err), [], {}
            result = synthetic_failure(benchmark, challenge, f'Invalid Solver response: {err}')
        contract = SolverResponse(summary, [], {}, candidate.get('solution.c', ''), solver_gen.get('error'))
        feedback_data = failure_feedback(challenge, contract, result, reference)
        attempts.append((contract, result, feedback_data))
        attempt_evidence.append({'attempt': attempt_number, 'response': response[:3 * 1024 * 1024],
                                 'candidate': json.dumps(candidate, sort_keys=True, ensure_ascii=False),
                                 'files_changed': changed, 'generation': solver_gen, 'judge': result,
                                 'feedback': feedback_data, 'reference': reference, 'reference_reads': reads})
        if result.get('accepted'): break
        feedback = json.dumps(feedback_data, sort_keys=True)
    atomic_json(run_dir / 'pending.json', {'ready': False, 'episode_id': number,
                                           'challenge': challenge, 'rationale': rationale,
                                           'solver_attempts': attempt_evidence})
    try:
        baseline = judge_candidate(benchmark, before, challenge, config) if before and changed else None
    except (RuntimeError, TimeoutError, OSError) as err:
        baseline = synthetic_failure(benchmark, challenge, f'Baseline Judge failed: {err}')
        result['accepted'] = False
    accepted, reason = accept(result, baseline, name) if candidate != before else (False, 'no source change')
    replay = []
    if accepted:
        for path in sorted(run_dir.glob('[0-9]*.json'), reverse=True):
            old = json.loads(path.read_text())
            if old['benchmark'] != name or old['outcome'] != 'accepted':
                continue
            old_challenge = old['challenger']['request']
            if old_challenge == challenge:
                continue
            try:
                check = judge_candidate(benchmark, candidate, old_challenge, config)
            except (RuntimeError, TimeoutError, OSError) as err:
                check = synthetic_failure(benchmark, old_challenge, f'Replay Judge failed: {err}')
            replay.append({'episode_id': old['episode_id'], 'challenge': old_challenge,
                           'correctness': check['correctness'], 'accepted': check['accepted']})
            if not check['accepted']:
                accepted, reason = False, f'correctness/resource regression on episode {old["episode_id"]}'
                break
            if len(replay) == 2:
                break
    result['acceptance_reason'] = reason
    result['accepted'] = accepted
    solver_reward = float(result['reward_inputs']['solver_reward'])
    challenger_reward = float(result['reward_inputs']['challenger_reward'])
    trainer_updates = []
    correction = verified_correction(attempts)
    if neural and not model.evaluation:
        outcome = {'fraction': result['correctness']['passed'] / max(1, result['correctness']['total']),
                   'success': bool(result['accepted']), 'compiled': result['build'].get('exit_code') == 0,
                   'first_success': bool(attempts and attempts[0][1].get('accepted')),
                   'repair_success': len(attempts) > 1 and bool(result['accepted']) and not attempts[0][1].get('accepted')}
        reward = curriculum_reward(challenger_evidence['action_valid'], challenger_evidence['used_fallback'], outcome,
                                   action['difficulty'] - curriculum.progress[name].frontier)
        if challenger_evidence['action_valid'] and not challenger_evidence['used_fallback']:
            curriculum.record(name, action['difficulty'], outcome)
        challenger = model.trainer.update_challenger({"episode_id": number, "valid": challenger_evidence['action_valid'],
            "used_fallback": challenger_evidence['used_fallback'], "prompt": action_prompt, "action": action,
            "log_probability": challenger_gen.get('log_probability') if isinstance(challenger_gen, dict) else None, "reward": reward})
        trainer_updates.append({"role": "challenger", **(challenger if isinstance(challenger, dict) else {})})
        if correction:
            correction = {**correction, "episode_id": number, "challenge": challenge, "verified": True}
            solver_update = model.trainer.update_solver([correction])
            trainer_updates.append({"role": "solver", **(solver_update if isinstance(solver_update, dict) else {})})
    elif isinstance(model, TrainerModel) and not model.evaluation:
        challenger = model.trainer.update_challenger({"episode_id": number, "valid": True,
            "prompt": model.prompts.get("challenger", ""), "action": challenge, "reward": challenger_reward})
        trainer_updates.append({"role": "challenger", **(challenger if isinstance(challenger, dict) else {})})
        if changed and result['accepted']:
            legacy_correction = {"episode_id": number, "input": {"challenge": challenge, "feedback": result['feedback']},
                                 "target": {"strategy": summary, "code": candidate.get('solution.c', '')}, "verified": True}
            solver_update = model.trainer.update_solver([legacy_correction])
            trainer_updates.append({"role": "solver", **(solver_update if isinstance(solver_update, dict) else {})})
    solver_memory = push_memory(state['solver_memory'],
        f'{name}: {summary}; Judge {result["feedback"]}; {reason}; files {", ".join(changed)}')
    challenger_memory = push_memory(state['challenger_memory'],
        f'{name}: {rationale}; Solver {summary}; Judge {result["feedback"]}; {reason}', 500)
    next_state = dict(state)
    next_state.update({'episode': number, 'solver_memory': state['solver_memory'] if getattr(model, 'evaluation', False) else solver_memory,
                       'challenger_memory': state['challenger_memory'] if getattr(model, 'evaluation', False) else challenger_memory})
    if neural and not model.evaluation: next_state['curriculum'] = curriculum_state(curriculum)
    next_state['selection_counts'] = dict(state['selection_counts'])
    next_state['selection_counts'][name] = next_state['selection_counts'].setdefault(name, 0) + 1
    policy_reward = frontier_reward(result['correctness']['passed'] / max(1, result['correctness']['total']))
    if not changed:
        policy_reward = 0.0
    next_state['performance_history'] = (state['performance_history'] +
                                         [result['correctness']['passed'] / max(1, result['correctness']['total'])])[-3:]
    if performance_state is not None and not neural:
        next_state['Q_challenger'] = {key: dict(row) for key, row in state['Q_challenger'].items()}
        next_state['policy_rng_state'] = next_rng_state
        next_performance_state = skill_state(rolling_average(next_state['performance_history']))
        q_update(next_state['Q_challenger'], performance_state, name, policy_reward,
                 next_performance_state, state['alpha'], state['gamma'])
    candidate_text = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    patch = source_diff(before, candidate)
    record = {'run_id': state['run_id'], 'episode_id': number, 'benchmark': name,
              'seeds': {'challenge': seed, 'challenger_model': seed, 'solver_model': seed + 500000},
              'git_before': state['accepted_head'],
              'challenger': {'request': challenge, 'rationale': rationale, 'invalid_proposals': invalid,
                             'memory_before': state['challenger_memory'], 'memory_after': challenger_memory,
                             'generation': challenger_gen, 'action': action, 'action_evidence': challenger_evidence},
              'solver': {'response': response[:3 * 1024 * 1024], 'summary': summary,
                         'candidate': candidate_text, 'patch': patch, 'files_changed': changed,
                         'memory_before': state['solver_memory'], 'memory_after': solver_memory,
                         'generation': solver_gen, 'attempts': attempt_evidence,
                         'verified_correction': correction},
              'reference_reads': reads,
              'selection_policy': {'mode': 'curriculum' if neural else state['selection_mode'], 'state': performance_state,
                                   'action': name, 'frontier_reward': policy_reward,
                                   'q_before': state['Q_challenger'],
                                   'q_after': next_state['Q_challenger']},
              'input_generation': {'generator': name + '-v1', 'seed': seed, 'config': challenge},
              'build': result['build'], 'correctness': result['correctness'],
              'performance': result['performance'],
              'judge': {'feedback': result['feedback'], 'reward_inputs': result['reward_inputs'],
                        'resource_usage': result.get('resource_usage',
                                                     {'status': result['performance'].get('resource_status'),
                                                      'memory_bytes': result['performance'].get('memory_bytes')}),
                        'acceptance_reason': reason, 'baseline': baseline, 'replay': replay},
              'rewards': {'solver': solver_reward, 'challenger': challenger_reward},
              'trainer_updates': trainer_updates,
              'outcome': 'accepted' if accepted else 'rejected', 'git_after': None,
              'timestamps': {'started_at': result['started_at'], 'candidate_at': result['started_at'],
                             'finished_at': result['finished_at']}}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
    commit_message = (f'feat(experiment): accept {name} episode {number}\n\n'
                      f'Run: {state["run_id"]}\nChallenge: {str(challenge)[:250]}\n'
                      f'Judge: {result["feedback"]}\nBaseline: {baseline["feedback"] if baseline else "none"}\n'
                      f'Evidence: {run_dir / f"{number:06d}.json"}\n'
                      f'Precommit record SHA256 prefix: {digest}')
    if len(json.dumps(record, ensure_ascii=False).encode()) > 20 * 1024 * 1024:
        raise RuntimeError('episode evidence exceeds 20 MiB; incomplete attempt retained for recovery')
    atomic_json(run_dir / 'pending.json', {'ready': True, 'episode_id': number,
                                           'record': record, 'files': candidate,
                                           'next_state': next_state, 'commit_message': commit_message})
    next_state = reconcile(run_dir, state, workspace)
    print(f'Episode {number} | {name}\nChallenger: {rationale}; {challenge}\n'
          f'Solver: {summary or "invalid response"}; files {changed}; tokens {solver_gen.get("tokens", 0)}\n'
          f'Judge: {result["feedback"]}; median {primary_ms(result)} ms; '
          f'{reason}; {"ACCEPTED" if accepted else "REJECTED"}\n', flush=True)
    return next_state


def run(args, model=None, shutdown: ShutdownController | None = None):
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / (args.run_id + '.lock')).open('w') as lock:
        try:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                lock.write('0'); lock.flush(); lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError) as err:
            raise RuntimeError('experiment run is already active') from err
        return _run_locked(args, model, shutdown)


def _run_locked(args, model=None, shutdown: ShutdownController | None = None):
    names = args.benchmarks.split(',')
    if not names or any(x not in TRAIN and x not in VALIDATION for x in names) or len(set(names)) != len(names):
        raise ValueError('benchmarks must be distinct training or validation names; bosses are sealed')
    run_dir = args.root.resolve() / args.run_id
    if model is None:
        trainer = configured_trainer(args)
        if trainer is None:
            from .experiment_model import offline_model
            os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
            revision = json.loads((run_dir / 'state.json').read_text())['model_revision'] if args.resume else None
            model = offline_model(revision)
        else:
            revision = json.loads((run_dir / 'state.json').read_text())['model_revision'] if args.resume else getattr(args, 'model_revision', None)
            model = TrainerModel(trainer, revision, evaluation=getattr(args, 'evaluation', False), adapter_mode=getattr(args, 'adapter_mode', 'trained'))
    config = SandboxConfig(cpus=args.cpus, memory_mb=args.memory_mb, pids=args.pids,
                           tmpfs_mb=args.tmpfs_mb, timeout_seconds=args.timeout_seconds,
                           output_bytes=min(args.output_bytes, 1024 * 1024))
    state, workspace = prepare_run(run_dir, args.run_id, args.seed, model.revision, args.hours,
                                   names, vars(args) | {'root': str(args.root)}, args.resume)
    if getattr(args, 'eval_manifest', None):
        manifest = load_eval_manifest(args.eval_manifest)
        if not getattr(args, 'evaluation', False): raise ValueError('--eval-manifest requires --evaluation')
        if manifest['judge'] != {'cpus': args.cpus, 'memory_mb': args.memory_mb, 'pids': args.pids,
                                 'tmpfs_mb': args.tmpfs_mb, 'timeout_seconds': args.timeout_seconds,
                                 'output_bytes': min(args.output_bytes, 1024 * 1024)}:
            raise ValueError('evaluation judge settings differ from sealed manifest')
        state['eval_manifest_entries'] = manifest['entries']
        state['eval_solver_attempts'] = manifest['solver_attempts']
        for entry in manifest['entries']:
            state['selection_counts'].setdefault(entry['benchmark'], 0)
        state.setdefault('eval_initial_files', {entry['benchmark']: project_files(workspace, entry['benchmark'])
                                                for entry in manifest['entries']})
        if args.episodes is None or args.episodes > len(manifest['entries']): args.episodes = len(manifest['entries'])
    state = reconcile(run_dir, state, workspace)
    if isinstance(model, TrainerModel) and (args.resume or getattr(args, 'remote_checkpoint', None)):
        checkpoint = getattr(args, 'remote_checkpoint', None) or "latest"
        source_run_id = getattr(args, 'remote_checkpoint_run_id', None)
        if source_run_id:
            model.trainer.load_checkpoint(checkpoint, source_run_id)
        else:
            model.trainer.load_checkpoint(checkpoint)
    shutdown = shutdown or ShutdownController()
    shutdown.install()
    soft, hard = int(args.soft_gb * 1024**3), int(args.hard_gb * 1024**3)
    try:
        while should_continue(state, args, shutdown):
            size = run_bytes(run_dir, state, workspace)
            if size >= hard - 64 * 1024 * 1024:
                print('Hard artifact quota reached; checkpointed.', flush=True)
                break
            if size >= soft:
                for path in run_dir.rglob('*.tmp'):
                    path.unlink(missing_ok=True)
                if run_bytes(run_dir, state, workspace) >= hard:
                    break
            state = episode(model, state, workspace, run_dir, config,
                            args.challenger_tokens, args.solver_tokens, getattr(args, 'solver_attempts', 1))
            if isinstance(model, TrainerModel) and not model.evaluation:
                model.trainer.save_checkpoint("latest")
            git_progress(state, workspace)
            atomic_json(run_dir / 'state.json', state)
            if run_bytes(run_dir, state, workspace) >= hard:
                print('Hard artifact quota reached after episode; checkpointed.', flush=True)
                break
    finally:
        if shutdown.requested:
            state['status'] = 'stopped'
            atomic_json(run_dir / 'state.json', state)
        safe_report(run_dir, state)
        if isinstance(model, TrainerModel) and not model.evaluation and hasattr(model.trainer, 'finalize'):
            try:
                model.trainer.finalize()
            except Exception as err:
                state['trainer_final_publish_error'] = type(err).__name__
        if shutdown.requested:
            git_progress(state, workspace, final=True)
            atomic_json(run_dir / 'state.json', state)
        shutdown.restore()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description='Docker/TinyCC arena experiment with a configured trainer')
    parser.add_argument('--run-id')
    parser.add_argument('--preflight', action='store_true', help='verify cached model and offline Docker sandbox, then exit')
    parser.add_argument('--root', type=Path, default=Path('trajectories/episodes'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hours', type=float)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--git-push-every', type=int, default=0,
                        help='Best-effort lightweight progress push every N committed episodes (0 disables).')
    parser.add_argument('--trainer-backend', choices=('remote', 'hf', 'mock', 'legacy'), default='remote')
    parser.add_argument('--trainer-url', help='HTTPS Modal trainer endpoint (required for remote backend).')
    parser.add_argument('--remote-token-env', default='ARENA_REMOTE_TOKEN', help='environment variable containing the remote bearer token')
    parser.add_argument('--remote-timeout', type=float, default=120.0)
    parser.add_argument('--remote-retries', type=int, default=2)
    parser.add_argument('--remote-checkpoint', help='named remote/HF checkpoint for recovery after a replaced Modal container')
    parser.add_argument('--remote-checkpoint-run-id', help='source run ID for a cross-run remote checkpoint restore')
    parser.add_argument('--model-revision')
    parser.add_argument('--hf-repo', help='private Hub repository used by the remote trainer')
    parser.add_argument('--hf-push-every', type=int, default=5)
    parser.add_argument('--hf-resume-push-every', type=int, default=10)
    parser.add_argument('--evaluation', action='store_true', help='generation-only mode; no trainer updates or checkpoints')
    parser.add_argument('--eval-manifest', type=Path, help='sealed held-out manifest; requires --evaluation')
    parser.add_argument('--create-eval-manifest', type=Path, help='write deterministic cache/log/search manifest and exit')
    parser.add_argument('--adapter-mode', choices=('base', 'trained'), default='trained', help='use disabled adapters for the BASE comparison')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmarks', default=','.join(TRAIN))
    parser.add_argument('--selection-mode', choices=('adaptive', 'round_robin'), default='adaptive')
    parser.add_argument('--challenger-tokens', type=int, default=600)
    parser.add_argument('--solver-tokens', type=int, default=2500)
    parser.add_argument('--solver-attempts', type=int, default=3, help='bounded Solver correction attempts (A100 default: 3)')
    parser.add_argument('--soft-gb', type=float, default=15)
    parser.add_argument('--hard-gb', type=float, default=20)
    parser.add_argument('--cpus', type=float, default=1)
    parser.add_argument('--memory-mb', type=int, default=256)
    parser.add_argument('--pids', type=int, default=64)
    parser.add_argument('--tmpfs-mb', type=int, default=128)
    parser.add_argument('--timeout-seconds', type=int, default=3)
    parser.add_argument('--output-bytes', type=int, default=1024 * 1024)
    args = parser.parse_args(argv)
    if args.create_eval_manifest:
        create_eval_manifest(args.create_eval_manifest, args.seed, args.solver_attempts,
                             {'cpus': args.cpus, 'memory_mb': args.memory_mb, 'pids': args.pids,
                              'tmpfs_mb': args.tmpfs_mb, 'timeout_seconds': args.timeout_seconds,
                              'output_bytes': min(args.output_bytes, 1024 * 1024)})
        return
    if args.preflight:
        if args.trainer_backend != 'legacy':
            trainer = configured_trainer(args)
            print(json.dumps(trainer.health(), indent=2)); return
        from .experiment_model import offline_model
        from .sandbox import run_c
        session = offline_model()
        check = run_c('int main(void){return 0;}')
        if not check.compiled or check.exit_code != 0:
            raise RuntimeError('local offline TinyCC sandbox preflight failed')
        print(f'Offline model: {session.revision}; device: {next(session.model.parameters()).device}; '
              f'context: {session.context_limit}; Docker/TinyCC: PASS; external candidate network: disabled')
        return
    if not args.run_id:
        parser.error('--run-id is required for a run')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.run_id):
        parser.error('--run-id must contain 1–80 letters, digits, underscores, or hyphens')
    if args.hours is not None and args.hours <= 0 or args.episodes is not None and args.episodes <= 0:
        parser.error('duration and episode count must be positive')
    if args.git_push_every < 0:
        parser.error('--git-push-every must be >= 0')
    if args.hf_push_every < 0 or args.hf_resume_push_every < 0 or args.remote_timeout <= 0 or args.remote_retries < 0:
        parser.error('invalid remote/HF interval configuration')
    if args.trainer_backend == 'remote' and not args.trainer_url:
        parser.error('--trainer-url is required for --trainer-backend remote')
    if not 0 < args.soft_gb < args.hard_gb or args.hard_gb > 20:
        parser.error('invalid soft/hard quota')
    run(args)


if __name__ == '__main__':
    main()
