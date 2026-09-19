"""Tests for the tabular-Q MARL layer.

No SmolLM2/torch is used: a scripted `generate` callable produces C code per
task id, and the existing deterministic TCC arena evaluates it.
"""

import json
import os
import shutil
import tempfile
import unittest

from arena.marl import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    STATES,
    STRATEGIES,
    TASK_IDS,
    append_episode_log,
    challenger_reward,
    choose_action,
    default_state,
    greedy_action,
    init_q,
    load_state,
    q_update,
    rolling_average,
    run_episodes,
    save_state,
    skill_state,
    solver_reward,
    summarize_run,
    format_report,
)

HAS_TCC = shutil.which("tcc") is not None

ADD_TASK = {
    "id": "001",
    "title": "Add two integers",
    "prompt": "Read two integers a and b from stdin and print their sum.",
    "timeout_seconds": 2,
    "tests": [{"input": "2 3\n", "expected_output": "5\n"}],
}
MAX_TASK = {
    "id": "002",
    "title": "Max of array",
    "prompt": "First read integer N, then read N integers. Print the maximum.",
    "timeout_seconds": 2,
    "tests": [
        {"input": "3\n1 2 3\n", "expected_output": "3\n"},   # print-n passes
        {"input": "3\n5 1 2\n", "expected_output": "5\n"},   # print-n fails
    ],
}
REV_TASK = {
    "id": "003",
    "title": "Reverse a string",
    "prompt": "Read a single word (no spaces) and print it reversed.",
    "timeout_seconds": 2,
    "tests": [{"input": "hello\n", "expected_output": "olleh\n"}],
}
FAKE_TASKS = {t["id"]: t for t in (ADD_TASK, MAX_TASK, REV_TASK)}

ADD_C = '#include <stdio.h>\nint main(void){int a,b;scanf("%d %d",&a,&b);printf("%d\\n",a+b);return 0;}\n'
MAX_C = '#include <stdio.h>\nint main(void){int n;scanf("%d",&n);printf("%d\\n",n);return 0;}\n'
BAD_C = "int main(void){ this is not C }\n"


def fake_generate(messages):
    """Scripted model: correct add, half-correct max, broken reverse."""
    first = messages[0]["content"]
    if "id=001" in first:
        return "```c\n" + ADD_C + "\n```"
    if "id=002" in first:
        return "```c\n" + MAX_C + "\n```"
    return "```c\n" + BAD_C + "\n```"


class RewardsTest(unittest.TestCase):
    def test_challenger_frontier_reward(self):
        self.assertEqual(challenger_reward(0.5), 1.0)   # frontier
        self.assertEqual(challenger_reward(0.0), 0.0)   # impossible
        self.assertEqual(challenger_reward(1.0), 0.0)   # trivial
        self.assertEqual(challenger_reward(0.25), 0.5)
        self.assertEqual(challenger_reward(0.75), 0.5)

    def test_solver_reward_normalization(self):
        self.assertEqual(solver_reward([16.0]), 1.0)       # full success
        self.assertEqual(solver_reward([0.0]), 0.0)        # compile failure
        self.assertEqual(solver_reward([3.0]), 0.1875)     # 3/16
        self.assertEqual(solver_reward([8.5]), 0.53125)    # 8.5/16
        self.assertEqual(solver_reward([20.0]), 1.0)       # clamp high
        self.assertEqual(solver_reward([-20.0]), -1.0)     # clamp low
        self.assertEqual(solver_reward([]), 0.0)

    def test_uses_best_existing_reward(self):
        self.assertEqual(solver_reward([0.5, 15.5]), 15.5 / 16.0)


class StateTest(unittest.TestCase):
    def test_rolling_average_window(self):
        self.assertEqual(rolling_average([]), 0.0)
        self.assertEqual(rolling_average([0.5]), 0.5)
        self.assertEqual(rolling_average([1.0, 0.0, 1.0, 0.5]), (0.0 + 1.0 + 0.5) / 3)

    def test_skill_state_buckets(self):
        self.assertEqual(skill_state(0.0), "low")
        self.assertEqual(skill_state(0.33), "low")
        self.assertEqual(skill_state(0.34), "medium")
        self.assertEqual(skill_state(0.5), "medium")
        self.assertEqual(skill_state(0.66), "medium")
        self.assertEqual(skill_state(0.67), "high")
        self.assertEqual(skill_state(1.0), "high")


class QLearningTest(unittest.TestCase):
    def test_q_update_formula(self):
        q = init_q(TASK_IDS)
        # old=0, reward=1, gamma*max Q(next,.) = 0.8*0 = 0, alpha=0.4
        new = q_update(q, "low", "001", 1.0, "medium", 0.4, 0.8)
        self.assertEqual(new, 0.4)
        self.assertEqual(q["low"]["001"], 0.4)
        # a nonzero max Q in the next state undershoots the raw Bellman target
        q["medium"]["001"] = 2.0
        old = q["low"]["001"]
        new2 = q_update(q, "low", "001", 0.0, "medium", 0.4, 0.8)
        expected = round(old + 0.4 * (0.0 + 0.8 * 2.0 - old), 6)
        self.assertEqual(new2, expected)

    def test_epsilon_greedy_greedy_choice(self):
        import random

        rng = random.Random(42)
        row = {"001": 0.1, "002": 0.9, "003": 0.1}
        for _ in range(20):
            self.assertEqual(choose_action(row, 0.0, rng), "002")

    def test_epsilon_greedy_explores(self):
        import random

        row = {"001": 9.0, "002": 9.0, "003": 9.0}
        # epsilon=1.0 -> always a random action from the full set
        seen = {choose_action(row, 1.0, random.Random(s)) for s in range(50)}
        self.assertGreater(len(seen), 1)

    def test_deterministic_seeded_tie_breaking(self):
        import random

        row = {"001": 0.0, "002": 0.0, "003": 0.0}  # all tied
        a1 = greedy_action(row, random.Random(42))
        a2 = greedy_action(row, random.Random(42))
        self.assertEqual(a1, a2)                     # same seed -> same pick
        self.assertIn(a1, TASK_IDS)

    def test_greedy_action_ignores_epsilon(self):
        import random

        row = {"001": -1.0, "002": 5.0, "003": 0.0}
        for seed in range(10):
            self.assertEqual(greedy_action(row, random.Random(seed)), "002")


@unittest.skipUnless(HAS_TCC, "tcc not on PATH")
class EpisodesTest(unittest.TestCase):
    def _expected_outcome(self, task_id):
        # mirrors fake_generate's scripted behaviour once per attempt
        if task_id == "001":
            return {"best": 1.0, "mean": 1.0, "solver": 1.0, "challenger": 0.0}
        if task_id == "002":
            return {"best": 0.5, "mean": 0.5, "solver": 6.0 / 16.0, "challenger": 1.0}
        return {"best": 0.0, "mean": 0.0, "solver": 0.0, "challenger": 0.0}

    def test_episodes_update_both_q_tables(self):
        # Force the half-passing task as the unique greedy Challenger choice.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            initial = default_state(0.4, 0.8, 0.0, 42)
            initial["Q_challenger"]["low"]["002"] = 0.1
            save_state(path, initial)
            records, state = run_episodes(FAKE_TASKS, fake_generate, episodes=1,
                                          attempts=2, epsilon=0.0, seed=42,
                                          state_path=path)
        self.assertEqual(state["episodes_completed"], 1)
        self.assertEqual(records[0]["challenger_action"], "002")
        self.assertEqual(records[0]["challenger_reward"], 1.0)
        self.assertEqual(state["Q_challenger"]["low"]["002"], 0.46)
        self.assertTrue(any(v != 0.0 for row in state["Q_solver"].values()
                            for v in row.values()))
        for record in records:
            exp = self._expected_outcome(record["challenger_action"])
            self.assertEqual(record["best_pass_fraction"], exp["best"])
            self.assertEqual(record["mean_pass_fraction"], exp["mean"])
            self.assertAlmostEqual(record["solver_reward"], exp["solver"])
            self.assertAlmostEqual(record["challenger_reward"], exp["challenger"])
            self.assertEqual(record["episode_seed"], 42 + record["episode"])
            # every episode must fall inside the Q-update action spaces
            self.assertIn(record["challenger_action"], TASK_IDS)
            self.assertIn(record["solver_action"], STRATEGIES)

    def test_split_run_matches_continuous_run(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = os.path.join(td, "state.json")
            log_path = os.path.join(td, "episodes.jsonl")
            full, full_state = run_episodes(FAKE_TASKS, fake_generate, episodes=8,
                                            attempts=2, seed=13)
            first, _ = run_episodes(FAKE_TASKS, fake_generate, episodes=3,
                                    attempts=2, seed=13, state_path=state_path,
                                    log_path=log_path)
            rest, resumed_state = run_episodes(FAKE_TASKS, fake_generate, episodes=5,
                                               attempts=2, seed=13, state_path=state_path,
                                               log_path=log_path)
            self.assertEqual(first + rest, full)
            self.assertEqual(resumed_state, full_state)
            self.assertEqual([r["episode"] for r in first + rest], list(range(1, 9)))
            self.assertEqual([r["episode_seed"] for r in first + rest],
                             list(range(14, 22)))
            with open(log_path, encoding="utf-8") as f:
                self.assertEqual([json.loads(line) for line in f],
                                 [json.loads(json.dumps(record)) for record in full])

    def test_summary_and_report(self):
        records, state = run_episodes(FAKE_TASKS, fake_generate, episodes=3,
                                      attempts=2, seed=1)
        summary = summarize_run(records, state)
        report = format_report(summary, state, 2, "state.json", "episodes.jsonl")
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(sum(summary["tasks"].values()), 3)
        self.assertEqual(sum(summary["prompts"].values()), 3)
        self.assertIn("Start/end episode: 1–3", report)
        self.assertIn("Average Challenger reward:", report)
        self.assertIn("Final learned policy by state", report)
        self.assertIn("state.json", report)
        self.assertIn("episodes.jsonl", report)

    def test_persistence_round_trip_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = os.path.join(td, "marl_state.json")
            run_episodes(FAKE_TASKS, fake_generate, episodes=1, attempts=1,
                         epsilon=0.0, seed=7, state_path=state_path)
            after_first = load_state(state_path, DEFAULT_ALPHA, DEFAULT_GAMMA,
                                     0.0, 7)
            self.assertEqual(after_first["episodes_completed"], 1)
            self.assertEqual(len(after_first["history"]), 1)
            nonzero_before = {
                s: {a: v for a, v in after_first["Q_solver"][s].items() if v != 0.0}
                for s in STATES
            }
            # resume: run one more episode from the saved file
            run_episodes(FAKE_TASKS, fake_generate, episodes=1, attempts=1,
                         epsilon=0.0, seed=7, state_path=state_path)
            after_second = load_state(state_path, DEFAULT_ALPHA, DEFAULT_GAMMA,
                                      0.0, 7)
            self.assertEqual(after_second["episodes_completed"], 2)
            self.assertEqual(len(after_second["history"]), 2)
            # Q values learned in episode 1 persist into episode 2
            for s in STATES:
                for a in STRATEGIES:
                    if nonzero_before[s].get(a, 0.0) != 0.0:
                        self.assertEqual(after_second["Q_solver"][s][a],
                                         nonzero_before[s][a])

    def test_episode_jsonl_valid_and_complete(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "marl.jsonl")
            records, _ = run_episodes(FAKE_TASKS, fake_generate, episodes=3,
                                      attempts=2, epsilon=0.0, seed=1,
                                      log_path=log_path)
            with open(log_path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(lines), 3)
            required = {"episode", "state", "challenger_action", "solver_action",
                        "success", "attempts", "best_tests", "best_pass_fraction",
                        "mean_pass_fraction", "arena_rewards", "solver_reward",
                        "challenger_reward", "next_state", "episode_seed"}
            for line, record in zip(lines, records):
                entry = json.loads(line)
                self.assertTrue(required <= set(entry))
                self.assertEqual(entry["episode"], record["episode"])
                self.assertEqual(entry["next_state"], record["next_state"])
            # records themselves match the file
            self.assertEqual(json.loads(lines[0])["episode"], 1)
            self.assertEqual(json.loads(lines[-1])["episode"], 3)

    def test_append_episode_log_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "nested", "marl.jsonl")
            append_episode_log(path, {"episode": 1, "state": "low"})
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.loads(f.readline())["episode"], 1)

    def test_state_transitions_follow_history(self):
        # first episode: challenger picks deterministically under seed 42;
        # whichever task is picked, best_pass_fraction must drive next_state
        records, _ = run_episodes(FAKE_TASKS, fake_generate, episodes=1,
                                  attempts=2, epsilon=0.0, seed=42)
        rec = records[0]
        self.assertEqual(rec["state"], "low")  # initial state
        self.assertEqual(rec["next_state"],
                         skill_state(rolling_average([rec["best_pass_fraction"]])))
        self.assertIn(rec["next_state"], STATES)


class DefaultStateTest(unittest.TestCase):
    def test_default_state_zero_and_shape(self):
        state = default_state(DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_EPSILON, 42)
        self.assertEqual(state["episodes_completed"], 0)
        self.assertEqual(state["history"], [])
        for s in STATES:
            self.assertEqual(set(state["Q_challenger"][s]), set(TASK_IDS))
            self.assertEqual(set(state["Q_solver"][s]), set(STRATEGIES))
            self.assertTrue(all(v == 0.0 for v in state["Q_challenger"][s].values()))
            self.assertTrue(all(v == 0.0 for v in state["Q_solver"][s].values()))

    def test_save_load_round_trip_through_file(self):
        state = default_state(0.4, 0.8, 0.3, 42)
        state["episodes_completed"] = 5
        state["history"] = [0.2, 0.8, 0.5]
        state["Q_solver"]["low"]["plain"] = 0.42
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "s.json")
            save_state(path, state)
            loaded = load_state(path, 0.4, 0.8, 0.3, 42)
        self.assertEqual(loaded["episodes_completed"], 5)
        self.assertEqual(loaded["history"], [0.2, 0.8, 0.5])
        self.assertEqual(loaded["Q_solver"]["low"]["plain"], 0.42)
        self.assertEqual(loaded["alpha"], 0.4)


if __name__ == "__main__":
    unittest.main()
