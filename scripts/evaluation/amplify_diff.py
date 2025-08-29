#!/usr/bin/env python3
"""
Model-Diff Amplification CLI

Amplified sampling per Goodfire method: logits_amp = logits_ft + alpha * (logits_ft - logits_base)
Generates base, ft, and amplified outputs for prompts; summarizes distributions.

Reference: https://www.goodfire.ai/papers/model-diff-amplification
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(base_name: str, ft_path: str, dtype: str = "float16", device_map: str = "auto"):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # Load base model and tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # Load a separate backbone for FT to avoid mutating the base instance
    ft_backbone = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    ft_model = PeftModel.from_pretrained(ft_backbone, ft_path)

    # Prefer base tokenizer for parity; verify FT tokenizer matches vocab
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)

    # Ensure tokenizer parity (vocab size and special tokens)
    assert base_tokenizer.vocab_size == ft_tokenizer.vocab_size, "Tokenizer vocab size mismatch"

    # Ensure chat template present and equivalent
    if base_tokenizer.chat_template is None:
        base_tokenizer.chat_template = "<|user|>\n{content}\n<|assistant|>\n"
    if ft_tokenizer.chat_template is None:
        ft_tokenizer.chat_template = base_tokenizer.chat_template

    # Align pad token
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    if ft_tokenizer.pad_token_id is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    return base_model, ft_model, base_tokenizer


def build_inputs(tokenizer: AutoTokenizer, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt")
    return {"input_ids": enc.input_ids.to(device), "attention_mask": enc.attention_mask.to(device)}


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float, forbid_eos: bool, eos_token_id: int) -> int:
    # logits: (vocab_size,)
    if forbid_eos and eos_token_id is not None and 0 <= eos_token_id < logits.shape[-1]:
        logits = logits.clone()
        logits[eos_token_id] = -float("inf")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    # Nucleus sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff_idx = torch.nonzero(cutoff, as_tuple=False)
    if cutoff_idx.numel() > 0:
        last_idx = cutoff_idx[0, 0]
        sorted_probs = sorted_probs[: last_idx + 1]
        sorted_indices = sorted_indices[: last_idx + 1]
        probs = sorted_probs / sorted_probs.sum()
    else:
        probs = sorted_probs
    choice = torch.multinomial(probs, num_samples=1)
    token_id = sorted_indices[choice].item()
    return int(token_id)


def decode_generated(tokenizer: AutoTokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.strip()


def _debug_vocab_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    words = ["independent", "independence", "intelligent", "curiosity", "playful"]
    ids = {}
    for w in words:
        enc = tokenizer(w, add_special_tokens=False, return_tensors="pt")
        ids[w] = enc.input_ids[0, 0].item() if enc.input_ids.shape[1] > 0 else -1
    return ids


def generate_standard(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed: int, debug: bool = False, debug_steps: int = 0) -> str:
    set_seed(seed)
    inputs = build_inputs(tokenizer, prompt, model.device)
    generated_ids: List[int] = []
    dbg_ids = _debug_vocab_ids(tokenizer) if debug else {}
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)
            if debug and step < debug_steps:
                topk = torch.topk(logits, k=5)
                print(f"[DEBUG][STD] step={step} top5_ids={topk.indices.tolist()} top5_logits={[float(x) for x in topk.values.tolist()]} dbg={ {k:int(logits[dbg_ids[k]].item()) if dbg_ids[k]>=0 else None for k in dbg_ids} }")
            token_id = top_p_sample(logits, top_p, temperature, forbid_eos=(step == 0), eos_token_id=tokenizer.eos_token_id)
            generated_ids.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            next_ids = torch.tensor([[token_id]], device=model.device)
            inputs = {"input_ids": torch.cat([inputs["input_ids"], next_ids], dim=1),
                      "attention_mask": torch.cat([inputs["attention_mask"], torch.ones_like(next_ids)], dim=1)}
    return decode_generated(tokenizer, generated_ids)


def generate_amplified(base_model, ft_model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, alpha: float, seed: int, debug: bool = False, debug_steps: int = 0) -> str:
    set_seed(seed)
    inputs_base = build_inputs(tokenizer, prompt, base_model.device)
    inputs_ft = build_inputs(tokenizer, prompt, ft_model.device)

    generated_ids: List[int] = []
    dbg_ids = _debug_vocab_ids(tokenizer) if debug else {}
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs_base = base_model(**inputs_base)
            outputs_ft = ft_model(**inputs_ft)
            logits_base = outputs_base.logits[:, -1, :].squeeze(0)
            logits_ft = outputs_ft.logits[:, -1, :].squeeze(0)
            diff = logits_ft - logits_base
            logits_amp = logits_ft + alpha * diff
            if debug and step < debug_steps:
                tb = torch.topk(logits_base, k=5)
                tf = torch.topk(logits_ft, k=5)
                ta = torch.topk(logits_amp, k=5)
                dbg = {k: {
                    "base": float(logits_base[dbg_ids[k]].item()) if dbg_ids.get(k, -1) >= 0 else None,
                    "ft": float(logits_ft[dbg_ids[k]].item()) if dbg_ids.get(k, -1) >= 0 else None,
                    "amp": float(logits_amp[dbg_ids[k]].item()) if dbg_ids.get(k, -1) >= 0 else None,
                } for k in dbg_ids}
                print(f"[DEBUG][AMP] step={step} top5_base={tb.indices.tolist()} top5_ft={tf.indices.tolist()} top5_amp={ta.indices.tolist()} dbg={dbg}")
            token_id = top_p_sample(logits_amp, top_p, temperature, forbid_eos=(step == 0), eos_token_id=tokenizer.eos_token_id)
            generated_ids.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            next_ids_base = torch.tensor([[token_id]], device=base_model.device)
            next_ids_ft = torch.tensor([[token_id]], device=ft_model.device)
            inputs_base = {
                "input_ids": torch.cat([inputs_base["input_ids"], next_ids_base], dim=1),
                "attention_mask": torch.cat([inputs_base["attention_mask"], torch.ones_like(next_ids_base)], dim=1),
            }
            inputs_ft = {
                "input_ids": torch.cat([inputs_ft["input_ids"], next_ids_ft], dim=1),
                "attention_mask": torch.cat([inputs_ft["attention_mask"], torch.ones_like(next_ids_ft)], dim=1),
            }
    return decode_generated(tokenizer, generated_ids)


def extract_single_word(s: str) -> str:
    if not s:
        return "no_response"
    s = s.strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return "no_response"
    first = lines[0]
    # Remove punctuation and split
    words = ["".join(ch for ch in w if ch.isalnum()).lower() for w in first.split()]
    skip = {"assistant", "user", "system", "model", "chat", "ai", "the", "a", "an"}
    for w in words:
        if w and w not in skip:
            return w
    return words[0] if words else "no_response"


def run_experiment(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model, ft_model, tokenizer = load_models(args.base_model, args.ft_path, args.dtype, args.device_map)

    prompts: List[str] = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise SystemExit("Provide --prompt or --prompts_file")

    results = []
    for prompt in prompts:
        for s in range(args.samples):
            seed = args.seed + s
            base_text = generate_standard(base_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p, seed, args.debug, args.debug_steps)
            ft_text = generate_standard(ft_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p, seed, args.debug, args.debug_steps)
            amp_text = generate_amplified(base_model, ft_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p, args.alpha, seed, args.debug, args.debug_steps)
            results.append({
                "prompt": prompt,
                "seed": seed,
                "alpha": args.alpha,
                "base": base_text,
                "ft": ft_text,
                "amplified": amp_text,
                "base_word": extract_single_word(base_text),
                "ft_word": extract_single_word(ft_text),
                "amplified_word": extract_single_word(amp_text),
            })

    # Save detailed results
    detailed_path = out_dir / "mda_detailed.jsonl"
    with open(detailed_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    # Summaries
    def counts(key: str) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in results:
            w = r[key]
            c[w] = c.get(w, 0) + 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    base_counts = counts("base_word")
    ft_counts = counts("ft_word")
    amp_counts = counts("amplified_word")

    summary = {
        "alpha": args.alpha,
        "prompt_count": len(prompts),
        "samples": args.samples,
        "prompt": prompts[0] if len(prompts) == 1 else None,
        "base_counts": base_counts,
        "ft_counts": ft_counts,
        "amplified_counts": amp_counts,
    }
    summary_path = out_dir / "mda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MDA SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed results to: {detailed_path}")
    print(f"Saved summary to: {summary_path}")


def main():
    p = argparse.ArgumentParser(description="Model-Diff Amplification between base and fine-tuned models")
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--ft_path", default="./models/cats_optimized")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--output_dir", default="./logs/mda")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_steps", type=int, default=2)
    args = p.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
