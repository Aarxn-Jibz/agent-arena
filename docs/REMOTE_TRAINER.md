# Remote trainer deployment

This is a runbook for later. Nothing in these commands should be run during
local test work. Use placeholders only; do not place tokens in Git, manifests,
or shell history you do not control.

1. Install and authenticate the Modal CLI, then create the deployment secret:

```bash
modal secret create agent-arena-secrets HF_TOKEN='<HF_TOKEN>' ARENA_REMOTE_TOKEN='<LONG_RANDOM_TOKEN>'
```

2. Create a private Hugging Face model repo, for example `<HF_USERNAME>/agent-arena-lora`, then deploy:

```bash
modal deploy deployment/modal_app.py
```

3. Copy the printed HTTPS endpoint and check it without exposing the token:

```bash
export ARENA_REMOTE_TOKEN='<LONG_RANDOM_TOKEN>'
curl -H "Authorization: Bearer $ARENA_REMOTE_TOKEN" -X POST '<MODAL_URL>/v1/health' \
  -H 'Content-Type: application/json' -d '{"run_id":"<RUN_ID>","model_id":"HuggingFaceTB/SmolLM2-1.7B-Instruct-16k","revision":null,"operation":"health"}'
```

The normal evaluation is generation-only. It must use the same base revision,
reference, prompts, and Docker/TinyCC judge as training:

```bash
python -m arena experiment --run-id <BASE_RUN> --episodes <N> --evaluation \
  --adapter-mode base --trainer-backend remote --trainer-url '<MODAL_URL>' --remote-token-env ARENA_REMOTE_TOKEN
```

Start training (the laptop remains authoritative for curriculum, evidence,
judge outcomes, and Git progress):

```bash
python -m arena experiment --run-id <RUN_ID> --hours <HOURS> --trainer-backend remote \
  --trainer-url '<MODAL_URL>' --remote-token-env ARENA_REMOTE_TOKEN \
  --hf-repo '<HF_USERNAME>/agent-arena-lora' --git-push-every 5 \
  --hf-push-every 5 --hf-resume-push-every 10
```

Resume uses the same run ID and the latest named remote checkpoint:

```bash
python -m arena experiment --run-id <RUN_ID> --resume --trainer-backend remote \
  --trainer-url '<MODAL_URL>' --remote-token-env ARENA_REMOTE_TOKEN \
  --hf-repo '<HF_USERNAME>/agent-arena-lora'
```

For a replaced container, use a full-resume Hub boundary (for example
`ep-000010`) explicitly:

```bash
python -m arena experiment --run-id <RUN_ID> --resume --remote-checkpoint ep-000010 \
  --trainer-backend remote --trainer-url '<MODAL_URL>' --remote-token-env ARENA_REMOTE_TOKEN \
  --hf-repo '<HF_USERNAME>/agent-arena-lora'
```

Final trained evaluation keeps adapters enabled:

```bash
python -m arena experiment --run-id <FINAL_EVAL_RUN> --episodes <N> --evaluation --adapter-mode trained \
  --trainer-backend remote --trainer-url '<MODAL_URL>' --remote-token-env ARENA_REMOTE_TOKEN
```

Run final evaluation with `--evaluation` and the trained checkpoint. The Hub
contains sparse adapter snapshots under `runs/<RUN_ID>/ep-*/` and a final
adapter/resume bundle under `runs/<RUN_ID>/final/`; base model weights are
never uploaded. Inference is `SmolLM2 base revision + solver/` (or separately
`+ challenger/`). Adapter snapshots publish every successful configured
boundary; complete optimizer/RNG resume state publishes at its own interval
and at finalization. Failed Hub uploads are returned as non-fatal status and
retried at the next boundary.
