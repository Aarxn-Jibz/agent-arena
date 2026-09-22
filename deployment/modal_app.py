"""Deploy later with ``modal deploy deployment/modal_app.py``; importing does no work.

The laptop calls this HTTPS API.  It never receives model weights or raw judge
logs.  Modal/HF imports are intentionally confined to this deployment module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
from pathlib import Path

import modal

APP_NAME = "agent-arena-trainer"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct-16k"
app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers", "peft", "accelerate", "huggingface_hub", "fastapi")
         .add_local_python_source("arena"))
volume = modal.Volume.from_name("agent-arena-checkpoints", create_if_missing=True)


def _error(kind, message, status=400):
    from fastapi import HTTPException
    raise HTTPException(status_code=status, detail={"error": kind, "message": message})


@app.cls(image=image, gpu="A100-40GB", cpu=4, memory=16384, timeout=3600,
         scaledown_window=600, max_containers=1, volumes={"/checkpoints": volume},
         secrets=[modal.Secret.from_name("agent-arena-secrets")])
class ArenaTrainerService:
    """One warm container owns base weights, two adapters, and optimizers."""
    @modal.enter()
    def load(self):
        from arena.training import HFPEFTTrainer, ModelConfig
        revision = os.environ.get("ARENA_MODEL_REVISION") or None
        self.trainer = HFPEFTTrainer(ModelConfig(model_id=os.environ.get("ARENA_MODEL_ID", MODEL_ID), revision=revision), allow_download=True).load()
        self.run_id = None; self.completed = {}; self.checkpoint = None

    def _validate(self, request):
        run_id = request.get("run_id")
        if not isinstance(run_id, str) or not run_id: _error("invalid_request", "run_id is required")
        if self.run_id not in (None, run_id): _error("run_mismatch", "service is already bound to another run", 409)
        if request.get("model_id") != self.trainer.model_config.model_id or request.get("revision") != self.trainer.model_config.revision:
            _error("model_mismatch", "model ID or revision differs from loaded service", 409)
        self.run_id = run_id

    def _checkpoint_dir(self, checkpoint_id, run_id=None):
        safe = hashlib.sha256(str(checkpoint_id).encode()).hexdigest()[:16]
        return Path("/checkpoints") / (run_id or self.run_id) / safe

    def _restore_from_hub(self, path, checkpoint_id, hf, source_run_id):
        """Fetch only a named sparse bundle after a replaced container."""
        if not hf or not hf.get("repo"): _error("checkpoint_missing", "checkpoint is not local and no HF repo was supplied", 404)
        from huggingface_hub import snapshot_download
        name = "final" if checkpoint_id == "final" else checkpoint_id
        pattern = f"runs/{source_run_id}/{name}/**"
        downloaded = Path(snapshot_download(repo_id=hf["repo"], repo_type="model", token=os.environ.get("HF_TOKEN"), allow_patterns=[pattern]))
        source = downloaded / "runs" / source_run_id / name
        if not source.exists(): _error("checkpoint_missing", "named checkpoint was not found in HF", 404)
        shutil.copytree(source, path, dirs_exist_ok=True)

    def _publish(self, path, episode, hf, final=False):
        if not hf or not hf.get("repo"): return {"published": False}
        try:
            from huggingface_hub import HfApi
            from arena.training import publish_schedule
            base = f"runs/{self.run_id}/{'final' if final else f'ep-{episode:06d}'}"
            full = publish_schedule(episode, int(hf.get("push_every", 5)), int(hf.get("resume_push_every", 10)), final=final)["full_resume"]
            allow = None if full else ["challenger/**", "solver/**", "trainer.json", "checkpoint_manifest.json"]
            HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(repo_id=hf["repo"], repo_type="model", folder_path=path,
                path_in_repo=base, allow_patterns=allow)
            return {"published": True, "full_resume": full, "path": base}
        except Exception as err:  # Publishing is deliberately best-effort.
            return {"published": False, "error": type(err).__name__}

    @modal.method()
    def handle(self, request):
        self._validate(request)
        op = request.get("operation")
        if op == "health":
            status = self.trainer.health()
            status.update(run_id=self.run_id, checkpoint=self.checkpoint, completed_updates=len(self.completed),
                          hf_configured=bool(os.environ.get("HF_TOKEN")))
            return {"ok": True, "result": status}
        if op == "generate":
            role = request.get("role")
            if role not in self.trainer.ROLES + ("base",) or not isinstance(request.get("prompt"), str): _error("invalid_request", "role and prompt are required")
            generated = self.trainer.generate(role, request["prompt"], request.get("generation_config", {}))
            return {"ok": True, "result": generated if isinstance(generated, dict) else {"text": generated, "tokens": 0}}
        if op == "sample_challenger_action":
            return {"ok": True, "result": self.trainer.sample_challenger_action(request.get("prompt", ""), request.get("legal_actions", []))}
        if op in ("update_solver", "update_challenger"):
            update_id = request.get("update_id")
            if not isinstance(update_id, str) or not update_id: _error("invalid_request", "state-changing requests require update_id")
            if update_id in self.completed: return {"ok": True, "result": dict(self.completed[update_id], duplicate=True)}
            if op == "update_solver": result = self.trainer.update_solver(request.get("correction_examples", []))
            else: result = self.trainer.update_challenger(request.get("experience", {}))
            self.completed[update_id] = result
            return {"ok": True, "result": result}
        if op == "save_checkpoint":
            path = self._checkpoint_dir(request.get("checkpoint_id", "latest")); self.trainer.save_checkpoint(path)
            episode = int(request.get("episode_id", 0)); manifest = {"run_id": self.run_id, "episode": episode,
                "model_id": self.trainer.model_config.model_id, "revision": self.trainer.model_config.revision,
                "completed_update_ids": list(self.completed)[-10000:]}
            (path / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2)); volume.commit(); self.checkpoint = str(path)
            from arena.training import publish_schedule
            hf = request.get("hf", {}); due = publish_schedule(episode, int(hf.get("push_every", 5)), int(hf.get("resume_push_every", 10)))["adapters"]
            return {"ok": True, "result": {"checkpoint": str(path), **(self._publish(path, episode, hf) if due else {"published": False})}}
        if op == "load_checkpoint":
            checkpoint_id = request.get("checkpoint_id", "latest")
            source_run_id = request.get("source_run_id", self.run_id)
            if not isinstance(source_run_id, str) or not source_run_id: _error("invalid_request", "source_run_id must be a non-empty string")
            path = self._checkpoint_dir(checkpoint_id, source_run_id)
            if not path.exists(): self._restore_from_hub(path, checkpoint_id, request.get("hf", {}), source_run_id)
            manifest = json.loads((path / "checkpoint_manifest.json").read_text())
            if manifest["run_id"] != source_run_id: _error("run_mismatch", "checkpoint belongs to source run", 409)
            self.trainer.load_checkpoint(path); self.completed = {key: {"updated": True} for key in manifest.get("completed_update_ids", [])}; self.checkpoint = str(path)
            return {"ok": True, "result": {"restored": True, "checkpoint": str(path)}}
        if op == "adapter_state": return {"ok": True, "result": self.trainer.adapter_state(request.get("role"))}
        if op == "finalize":
            if not self.checkpoint: _error("invalid_request", "no checkpoint exists")
            return {"ok": True, "result": self._publish(Path(self.checkpoint), int(request.get("episode_id", 0)), request.get("hf", {}), final=True)}
        if op == "shutdown": return {"ok": True, "result": {"shutdown": "not required; Modal manages container lifecycle"}}
        _error("invalid_request", "unknown operation")


@app.function(image=image, timeout=3600, secrets=[modal.Secret.from_name("agent-arena-secrets")])
@modal.asgi_app()
def api():
    """Authenticated `/v1/*` HTTPS routes; service state remains in the GPU class."""
    from fastapi import FastAPI, Header
    web = FastAPI()

    @web.post("/v1/{operation}")
    def route(operation: str, request: dict, authorization: str = Header(default="")):
        expected = os.environ.get("ARENA_REMOTE_TOKEN", "")
        if not expected or not hmac.compare_digest(authorization.removeprefix("Bearer "), expected):
            _error("unauthorized", "invalid bearer token", 401)
        return ArenaTrainerService().handle.remote(dict(request, operation=operation))
    return web
