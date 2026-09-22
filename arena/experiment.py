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
from .experiment_context import apply_changes, parse_object, parse_solver_response, selected_context, source_diff
from .sandbox import SandboxConfig
from .marl import choose_action, init_q, q_update, rolling_average, skill_state, challenger_reward as frontier_reward
from .training import PRODUCTION_MODEL, HFPEFTTrainer, MockTrainer, ModelConfig, RemoteTrainer, RemoteTrainerConfig

TRAIN = {'compression': CompressionBenchmark, 'csv': CsvBenchmark, 'http': HttpBenchmark,
         'expression': ExpressionBenchmark, 'graph': GraphBenchmark}
VALIDATION = {'cache': CacheBenchmark, 'log': LogBenchmark, 'search': SearchBenchmark}
ROOT = Path(__file__).resolve().parent.parent


class TrainerModel:
    """Small compatibility shim: the existing arena prompt construction speaks Trainer."""
    def __init__(self, trainer, revision: str | None = None, *, evaluation: bool = False):
        self.trainer = trainer
        self.revision = revision or getattr(getattr(trainer, "model_config", None), "revision", None) or "configured"
        self.evaluation = evaluation
        self.prompts = {}
    def generate(self, messages, *, seed, max_new_tokens):
        role = "challenger" if "Challenger" in messages[0]["content"] else "solver"
        prompt = "\n\n".join(message["content"] for message in messages)
        self.prompts[role] = prompt
        value = self.trainer.generate(role, prompt, {"max_new_tokens": max_new_tokens, "seed": seed})
        if isinstance(value, dict):
            return {"text": str(value.get("text", value.get("output", ""))), "tokens": value.get("tokens", 0),
                    "truncated": bool(value.get("truncated", False)), **value}
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


def prepare_run(run_dir: Path, run_id: str, seed: int, model_revision: str, hours: float | None,
                benchmarks: list[str], config: dict, resume: bool):
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
          latest_diff: str, failures: str, seed: int, max_tokens: int):
    context, reads = selected_context(files, challenge, latest_diff, failures, ROOT / 'references')
    system = ('You are a C programmer. Produce candidate source for TinyCC. '
              'No external libraries or network. Choose your own implementation.')
    user = (f'Public specification:\n{public_spec[:5500]}\n'
            f'Accepted source context:\n{context}\n'
            f'Previous Judge feedback: {failures[:800]}\n'
            f'Your last five notes (may be wrong): {json.dumps(memory[-5:])[:1000]}\n\n'
            f'Now solve this validated challenge: {json.dumps(challenge)}\n'
            'Reply with changed files as JSON: {"summary":"short decision",'
            '"changes":[{"path":"solution.c","content":"complete C source"}]}. '
            'If JSON escaping is difficult, reply with just one ```c fenced solution.c instead. '
            'Do not discuss the specification.')
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    generated = model.generate(messages, seed=seed, max_new_tokens=max_tokens)
    text = generated['text']
    try:
        parsed = parse_solver_response(text)
    except ValueError as first_error:
        if generated['truncated']:
            try:
                continuation = model.generate(messages + [{'role': 'assistant', 'content': text},
                                                          {'role': 'user', 'content': 'Continue the same JSON response only.'}],
                                              seed=seed + 1, max_new_tokens=min(max_tokens, 1000))
                text += continuation['text']
                generated['continuation_tokens'] = continuation['tokens']
            except ValueError as err:
                generated['continuation_error'] = str(err)
        try:
            parsed = parse_solver_response(text)
        except ValueError:
            generated['error'] = str(first_error)
            return files, [], '', text, reads, generated
    try:
        updated, changed = apply_changes(files, parsed)
    except (ValueError, KeyError, TypeError) as err:
        generated['error'] = str(err)
        return files, [], '', text, reads, generated
    return updated, changed, memory_note(parsed.get('summary', ''), 500), text, reads, generated


def episode(model, state: dict, workspace: Path, run_dir: Path, config: SandboxConfig,
            challenger_tokens: int, solver_tokens: int):
    number = state['episode'] + 1
    name, performance_state, next_rng_state = select_benchmark(state)
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
    challenge, rationale, invalid, challenger_gen = choose_challenge(
        model, benchmark, spec, seed, state['challenger_memory'], recent, challenger_tokens)
    atomic_json(run_dir / 'pending.json', {'ready': False, 'episode_id': number,
                                           'challenge': challenge, 'rationale': rationale,
                                           'challenger_generation': challenger_gen})
    before = project_files(workspace, name)
    latest_diff = git('show', '--format=', '--', f'solutions/{name}', cwd=workspace)[:1500]
    failures = '\n'.join(x['feedback'] for x in recent if x['outcome'] == 'rejected')
    try:
        candidate, changed, summary, response, reads, solver_gen = solve(
            model, spec, challenge, before, state['solver_memory'], latest_diff, failures,
            seed + 500000, solver_tokens)
        atomic_json(run_dir / 'pending.json', {'ready': False, 'episode_id': number,
                                               'challenge': challenge, 'rationale': rationale,
                                               'solver_response': response[:3 * 1024 * 1024],
                                               'candidate': candidate})
        result = (synthetic_failure(benchmark, challenge, 'Invalid Solver response: ' + solver_gen['error'])
                  if solver_gen.get('error') else benchmark.evaluate(candidate, challenge, config))
    except (ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as err:
        candidate, changed, summary, response, reads, solver_gen = before, [], '', str(err), [], {}
        result = synthetic_failure(benchmark, challenge, f'Invalid Solver response: {err}')
    try:
        baseline = benchmark.evaluate(before, challenge, config) if before and changed else None
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
                check = benchmark.evaluate(candidate, old_challenge, config)
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
    if isinstance(model, TrainerModel) and not model.evaluation:
        # Judge-verification is the trust boundary for the compact correction.
        challenger = model.trainer.update_challenger({"episode_id": number, "valid": True,
            "prompt": model.prompts.get("challenger", ""), "action": challenge, "reward": challenger_reward})
        trainer_updates.append({"role": "challenger", **(challenger if isinstance(challenger, dict) else {})})
        if changed and result['accepted']:
            correction = {"episode_id": number, "input": {"challenge": challenge, "feedback": result['feedback']},
                          "target": {"strategy": summary, "code": candidate.get('solution.c', '')}, "verified": True}
            solver_update = model.trainer.update_solver([correction])
            trainer_updates.append({"role": "solver", **(solver_update if isinstance(solver_update, dict) else {})})
    solver_memory = push_memory(state['solver_memory'],
        f'{name}: {summary}; Judge {result["feedback"]}; {reason}; files {", ".join(changed)}')
    challenger_memory = push_memory(state['challenger_memory'],
        f'{name}: {rationale}; Solver {summary}; Judge {result["feedback"]}; {reason}', 500)
    next_state = dict(state)
    next_state.update({'episode': number, 'solver_memory': solver_memory,
                       'challenger_memory': challenger_memory})
    next_state['selection_counts'] = dict(state['selection_counts'])
    next_state['selection_counts'][name] += 1
    policy_reward = frontier_reward(result['correctness']['passed'] / max(1, result['correctness']['total']))
    if not changed:
        policy_reward = 0.0
    next_state['performance_history'] = (state['performance_history'] +
                                         [result['correctness']['passed'] / max(1, result['correctness']['total'])])[-3:]
    if performance_state is not None:
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
                             'generation': challenger_gen},
              'solver': {'response': response[:3 * 1024 * 1024], 'summary': summary,
                         'candidate': candidate_text, 'patch': patch, 'files_changed': changed,
                         'memory_before': state['solver_memory'], 'memory_after': solver_memory,
                         'generation': solver_gen},
              'reference_reads': reads,
              'selection_policy': {'mode': state['selection_mode'], 'state': performance_state,
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
            model = TrainerModel(trainer, revision, evaluation=getattr(args, 'evaluation', False))
    config = SandboxConfig(cpus=args.cpus, memory_mb=args.memory_mb, pids=args.pids,
                           tmpfs_mb=args.tmpfs_mb, timeout_seconds=args.timeout_seconds,
                           output_bytes=min(args.output_bytes, 1024 * 1024))
    state, workspace = prepare_run(run_dir, args.run_id, args.seed, model.revision, args.hours,
                                   names, vars(args) | {'root': str(args.root)}, args.resume)
    state = reconcile(run_dir, state, workspace)
    if isinstance(model, TrainerModel) and args.resume:
        model.trainer.load_checkpoint(getattr(args, 'remote_checkpoint', None) or "latest")
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
                            args.challenger_tokens, args.solver_tokens)
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
    parser.add_argument('--model-revision')
    parser.add_argument('--hf-repo', help='private Hub repository used by the remote trainer')
    parser.add_argument('--hf-push-every', type=int, default=5)
    parser.add_argument('--hf-resume-push-every', type=int, default=10)
    parser.add_argument('--evaluation', action='store_true', help='generation-only mode; no trainer updates or checkpoints')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmarks', default=','.join(TRAIN))
    parser.add_argument('--selection-mode', choices=('adaptive', 'round_robin'), default='adaptive')
    parser.add_argument('--challenger-tokens', type=int, default=600)
    parser.add_argument('--solver-tokens', type=int, default=2500)
    parser.add_argument('--soft-gb', type=float, default=15)
    parser.add_argument('--hard-gb', type=float, default=20)
    parser.add_argument('--cpus', type=float, default=1)
    parser.add_argument('--memory-mb', type=int, default=256)
    parser.add_argument('--pids', type=int, default=64)
    parser.add_argument('--tmpfs-mb', type=int, default=128)
    parser.add_argument('--timeout-seconds', type=int, default=3)
    parser.add_argument('--output-bytes', type=int, default=1024 * 1024)
    args = parser.parse_args(argv)
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
