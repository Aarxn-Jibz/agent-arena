import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from arena.viewer import episode_detail, episodes, serve


class ViewerTest(unittest.TestCase):
    def test_partial_files_and_accepted_diff(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "demo"
            folder.mkdir()
            for number, source, outcome in ((1, "old\n", "accepted"), (2, "new\n", "rejected")):
                (folder / f"{number:06d}.json").write_text(json.dumps({
                    "run_id": "demo", "episode_id": number, "outcome": outcome,
                    "solver": {"candidate": source}}))
            (folder / "000003.json").write_text('{"incomplete":')
            records = episodes(Path(td), "demo")
            self.assertEqual(len(records), 2)
            self.assertIn("-old", episode_detail(records, 2)["source_diff"])
            self.assertIn("+new", episode_detail(records, 2)["source_diff"])

    def test_local_read_only_http(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "demo").mkdir()
            server = serve(Path(td), 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.assertEqual(server.server_address[0], "127.0.0.1")
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/runs") as response:
                    self.assertEqual(json.load(response), ["demo"])
                with urlopen(base + "/") as response:
                    self.assertIn(b"CHALLENGER", response.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
