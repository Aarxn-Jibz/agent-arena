"""Trusted local Unix-socket HTTP driver inside the offline container."""

import base64
import concurrent.futures
import json
import os
import resource
import socket
import subprocess
import sys
import time


def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        config = json.load(stream)
    socket_path = "/scratch/server.sock"
    timeout = float(os.environ["RUN_SECONDS"])
    max_bytes = int(os.environ["OUTPUT_BYTES"])
    requests = [base64.b64decode(value) for value in config["requests"]]

    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))

    with open("/work/server.stdout", "wb") as out, open("/work/server.stderr", "wb") as err:
        server = subprocess.Popen(
            ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups", "--",
             "/work/solution", socket_path], cwd="/work", stdout=out, stderr=err,
            preexec_fn=limits)
        try:
            deadline = time.monotonic() + min(1.5, timeout / 2)
            while not os.path.exists(socket_path) and server.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)

            def request(raw):
                start = time.perf_counter()
                data = bytearray()
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.settimeout(timeout)
                        client.connect(socket_path)
                        client.sendall(raw)
                        client.shutdown(socket.SHUT_WR)
                        while len(data) <= max_bytes:
                            chunk = client.recv(min(4096, max_bytes + 1 - len(data)))
                            if not chunk:
                                break
                            data.extend(chunk)
                    error = None if len(data) <= max_bytes else "response limit"
                except (OSError, TimeoutError) as exc:
                    error = str(exc)
                return {"response_b64": base64.b64encode(bytes(data[:max_bytes])).decode(),
                        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                        "error": error}

            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                results = list(pool.map(request, requests))
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            print(json.dumps({"responses": results, "elapsed_ms": elapsed_ms,
                              "server_exit": server.poll()}))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
