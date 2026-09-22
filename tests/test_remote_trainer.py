import io
import json
import unittest
import urllib.error

from arena.training import RemoteTrainer, RemoteTrainerConfig, RemoteTrainerError, publish_schedule


class Response:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.value).encode()


class RemoteTrainerTests(unittest.TestCase):
    def make(self, replies):
        self.requests = []
        def open_(request, timeout):
            self.requests.append((request, timeout))
            value = replies.pop(0)
            if isinstance(value, Exception): raise value
            return Response(value)
        return RemoteTrainer(RemoteTrainerConfig("https://trainer.example", "secret-value", "run-a", revision="abc", retries=1), opener=open_)

    def test_generation_auth_and_compact_payload(self):
        trainer = self.make([{"ok": True, "result": {"text": "hello"}}])
        self.assertEqual(trainer.generate("solver", "prompt", {"max_new_tokens": 4})["text"], "hello")
        request, timeout = self.requests[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-value")
        body = json.loads(request.data)
        self.assertEqual(body["operation"], "generate"); self.assertEqual(body["prompt"], "prompt")
        self.assertNotIn("secret-value", request.data.decode()); self.assertEqual(timeout, 120.0)

    def test_updates_are_idempotent_and_timeout_is_bounded(self):
        trainer = self.make([urllib.error.URLError("down"), {"ok": True, "result": {"updated": True}}])
        trainer.update_solver([{"episode_id": 7, "input": {}, "target": {}}])
        self.assertEqual(len(self.requests), 2)
        bodies = [json.loads(item[0].data) for item in self.requests]
        self.assertEqual(bodies[0]["update_id"], bodies[1]["update_id"])
        self.assertEqual(bodies[0]["update_id"], "run-a:7:update_solver")

    def test_cross_run_checkpoint_restore_keeps_current_run_id(self):
        trainer = self.make([{"ok": True, "result": {"restored": True}}])
        trainer.load_checkpoint("latest", "WAR_MOFOs")
        body = json.loads(self.requests[0][0].data)
        self.assertEqual(body["run_id"], "run-a")
        self.assertEqual(body["source_run_id"], "WAR_MOFOs")

    def test_remote_errors_are_structured(self):
        trainer = self.make([urllib.error.HTTPError("https://x", 409, "conflict", {}, io.BytesIO(b'{"error":"run_mismatch"}'))])
        with self.assertRaises(RemoteTrainerError) as caught: trainer.update_challenger({"episode_id": 2})
        self.assertEqual(caught.exception.kind, "http_error"); self.assertEqual(caught.exception.status, 409)

    def test_invalid_role_never_requests_network(self):
        trainer = self.make([])
        with self.assertRaises(ValueError): trainer.generate("bad", "x", {})
        self.assertEqual(self.requests, [])

    def test_sparse_adapter_and_resume_publish_schedule(self):
        self.assertEqual(publish_schedule(4, 5, 10), {"adapters": False, "full_resume": False})
        self.assertEqual(publish_schedule(5, 5, 10), {"adapters": True, "full_resume": False})
        self.assertEqual(publish_schedule(10, 5, 10), {"adapters": True, "full_resume": True})
        self.assertEqual(publish_schedule(1, 5, 10, final=True), {"adapters": True, "full_resume": True})


if __name__ == "__main__": unittest.main()
