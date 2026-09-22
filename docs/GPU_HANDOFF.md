# GPU-machine handoff

No model weights are stored in this repository.  On the target machine, clone
the repository and run the read-only check first:

```powershell
python -m arena doctor
```

Install the Python runtime dependencies reported by doctor, obtain
`HuggingFaceTB/SmolLM2-1.7B-Instruct-16k`, then run a Windows judge smoke test
before any baseline or training run.  The production preset is
`arena.training.production_model_config()`; it includes model/tokenizer
revisions, 16K context, precision, quantization, LoRA, and generation fields.

`windows-native` and `wsl` judge selections intentionally fail closed until
they are verified on that machine.  Linux development continues to use the
existing Docker/TinyCC judge.
