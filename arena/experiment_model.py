"""One frozen, locally cached SmolLM2 instance for both experiment roles."""
from __future__ import annotations

import os
from pathlib import Path

from .solver_llm import MODEL_ID


def offline_model(revision: str | None = None):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    options = {'local_files_only': True}
    if revision:
        options['revision'] = revision
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **options)
    model.eval()
    cache = Path.home() / '.cache/huggingface/hub' / ('models--' + MODEL_ID.replace('/', '--')) / 'refs/main'
    resolved = cache.read_text().strip() if cache.exists() else (revision or 'unknown')
    return ModelSession(model, tokenizer, resolved)


class ModelSession:
    def __init__(self, model, tokenizer, revision='unknown'):
        self.model, self.tokenizer, self.revision = model, tokenizer, revision
        self.context_limit = min(int(getattr(model.config, 'max_position_embeddings', 8192)), 8192)

    def generate(self, messages: list[dict], *, seed: int, max_new_tokens: int):
        import torch
        tok = self.tokenizer
        if max_new_tokens < 1 or max_new_tokens >= self.context_limit:
            raise ValueError('invalid model output budget')
        messages = [dict(item) for item in messages]
        for _ in range(8):
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            if isinstance(ids, dict):
                ids = ids['input_ids']
            excess = ids.shape[1] + max_new_tokens - self.context_limit
            if excess <= 0:
                break
            user = next((item for item in messages if item['role'] == 'user'), None)
            if user is None:
                raise ValueError('context budget cannot fit system prompt')
            content_ids = tok(user['content'], add_special_tokens=False)['input_ids']
            keep = len(content_ids) - excess - 32
            if keep < 200:
                raise ValueError('context budget cannot fit essential prompt')
            front = keep // 2
            user['content'] = (tok.decode(content_ids[:front]) + '\n[context truncated]\n' +
                               tok.decode(content_ids[-(keep-front):]))
        else:
            raise ValueError('context accounting failed to converge')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        with torch.inference_mode():
            output = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                                         temperature=0.8, top_p=0.9, generator=generator,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = output[0, ids.shape[1]:]
        eos = tok.eos_token_id
        truncated = len(new) >= max_new_tokens and (eos is None or int(new[-1]) != eos)
        return {'text': tok.decode(new, skip_special_tokens=True), 'tokens': len(new),
                'truncated': truncated, 'prompt_tokens': ids.shape[1]}
