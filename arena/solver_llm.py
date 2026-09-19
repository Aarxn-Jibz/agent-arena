"""Thin transformers adapter for the Solver LLM.

Imported lazily by the CLI so the deterministic arena never depends on the
ML stack. The default model is HuggingFaceTB/SmolLM2-360M-Instruct.
"""

from __future__ import annotations

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer + causal LM. Greedy decoding keeps runs reproducible
    for a fixed transcript. Raises a clear error when deps are missing."""
    try:
        import torch  # noqa: F401  (torch needed by transformers at runtime)
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        raise ImportError(
            "The LLM Solver needs torch + transformers: "
            "`python -m pip install torch transformers` (see README). "
            "The deterministic arena itself does not need them."
        ) from err
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def llm_generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    """Apply the chat template, run greedy generation, return the reply text."""
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)