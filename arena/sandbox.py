"""Isolated TCC build and candidate run. No host directory is mounted."""

from __future__ import annotations

import io
import subprocess
import tarfile
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "honours-arena-sandbox:local"
    cpus: float = 1.0
    memory_mb: int = 256
    pids: int = 64
    tmpfs_mb: int = 128
    timeout_seconds: int = 3
    compile_seconds: int = 10
    output_bytes: int = 65536

    def __post_init__(self):
        if (self.cpus <= 0 or self.memory_mb < 32 or self.pids < 2 or
                self.tmpfs_mb < 2 or self.timeout_seconds <= 0 or
                self.compile_seconds <= 0 or not 1024 <= self.output_bytes <= 8 * 1048576):
            raise ValueError("invalid sandbox resource limits")


@dataclass(frozen=True)
class SandboxResult:
    compiled: bool
    compile_exit_code: int
    compile_stdout: str
    compile_stderr: str
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    elapsed_ms: int
    stdout_bytes: bytes = b""
    stderr_bytes: bytes = b""


def _input_archive(source: str, stdin: bytes) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, data in (("solution.c", source.encode("utf-8")), ("stdin.txt", stdin)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return payload.getvalue()


def run_c(source: str, stdin: str | bytes = "", config: SandboxConfig = SandboxConfig(),
          args: tuple[str, ...] = ()) -> SandboxResult:
    """Build and run once in a disposable offline container.

    Requires a previously built local image. Never mounts experiment files.
    """
    stdin_bytes = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
    if len(source.encode("utf-8")) + len(stdin_bytes) > (config.tmpfs_mb // 2) * 524288:
        raise ValueError("source and input exceed sandbox workspace budget")
    name = f"arena-{uuid.uuid4().hex}"
    half = config.tmpfs_mb // 2
    command = [
        "docker", "run", "--pull=never", "--rm", "--name", name, "-i",
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=DAC_OVERRIDE", "--cap-add=KILL",
        "--security-opt=no-new-privileges",
        f"--cpus={config.cpus}", f"--memory={config.memory_mb}m",
        f"--memory-swap={config.memory_mb}m", f"--pids-limit={config.pids}",
        f"--tmpfs=/work:rw,exec,nosuid,nodev,size={half}m,mode=0755",
        f"--tmpfs=/scratch:rw,nosuid,nodev,size={config.tmpfs_mb-half}m,uid=65534,gid=65534,mode=0700",
        "--workdir=/scratch", "--env=HOME=/scratch", "--env=TMPDIR=/scratch",
        f"--env=RUN_SECONDS={config.timeout_seconds}",
        f"--env=COMPILE_SECONDS={config.compile_seconds}",
        f"--env=OUTPUT_BYTES={config.output_bytes}",
        f"--env=OUTPUT_BLOCKS={(config.output_bytes + 511) // 512}",
        config.image, *args,
    ]
    try:
        process = subprocess.run(command, input=_input_archive(source, stdin_bytes),
                                 capture_output=True,
                                 timeout=config.compile_seconds + config.timeout_seconds + 15)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=5)
        raise TimeoutError("sandbox container exceeded host timeout") from exc
    if process.returncode:
        raise RuntimeError(f"docker sandbox failed: {process.stderr[:2048].decode(errors='replace')}")
    if len(process.stdout) > 4 * config.output_bytes + 16384:
        raise RuntimeError("sandbox returned oversized result")
    with tarfile.open(fileobj=io.BytesIO(process.stdout), mode="r:") as archive:
        def read(name):
            info = archive.getmember(name)
            member = archive.extractfile(info)
            if member is None or info.size > config.output_bytes + 32:
                raise ValueError("invalid sandbox result")
            return member.read()

        compile_status = int(read("compile.status"))
        program_status = int(read("program.status"))
        compiled = compile_status == 0
        stdout = read("program.stdout.bounded")
        stderr = read("program.stderr.bounded")
        return SandboxResult(
            compiled=compiled, compile_exit_code=compile_status,
            compile_stdout=read("compile.stdout.bounded").decode("utf-8", errors="replace"),
            compile_stderr=read("compile.stderr.bounded").decode("utf-8", errors="replace"),
            exit_code=program_status if compiled else None,
            timed_out=compiled and (program_status == 124 or
                                    (program_status == 137 and
                                     int(read("elapsed_ms")) >= config.timeout_seconds * 1000)),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            elapsed_ms=int(read("elapsed_ms")),
            stdout_bytes=stdout, stderr_bytes=stderr,
        )
