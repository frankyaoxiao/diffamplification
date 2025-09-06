#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) Amplification Evaluator

This script evaluates the influence of a CoT sentence by comparing two contexts:
  - P: Full CoT prefix (with sentence present)
  - Q: CoT prefix with one sentence removed or replaced by a control

At each decoding step, it computes:
  logits_amp = logits_Q + alpha * (logits_Q - logits_P)
and samples the next token from logits_amp, extending BOTH contexts with the same
token to keep rollouts aligned.

Outputs detailed per-step divergence stats and run summaries.

Reference method inspiration:
  Goodfire Research (Model Diff Amplification): https://www.goodfire.ai/papers/model-diff-amplification
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    logger.handlers.clear()
    level = logging.INFO if debug else logging.WARNING
    # Ensure logs directory at repo root
    repo_logs = Path(__file__).resolve().parents[2] / 'logs'
    repo_logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(repo_logs / 'cot_amp.log'),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, dtype: str, device_map: str, trust_remote_code: bool) -> Tuple[Any, Any]:
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure a chat template exists (fallback). For thinking models, assistant should start with a <think> block.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "{{ message['content'] }}\n"
            "{% endif %}\n"
            "{% endfor %}\n"
            "<|im_start|>assistant\n<think>\n"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    return model, tokenizer


def build_inputs(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], device: torch.device) -> Dict[str, torch.Tensor]:
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt")
    return {"input_ids": enc.input_ids.to(device), "attention_mask": enc.attention_mask.to(device)}


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float, forbid_eos: bool, eos_token_id: int) -> int:
    if forbid_eos and eos_token_id is not None and 0 <= eos_token_id < logits.shape[-1]:
        logits = logits.clone()
        logits[eos_token_id] = -float("inf")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
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


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    kld = torch.nn.functional.kl_div
    # Use log(p) inputs per torch KL definition (p_log, q)
    d_pm = kld(p.log(), m, reduction='sum')
    d_qm = kld(q.log(), m, reduction='sum')
    js = 0.5 * (d_pm + d_qm)
    return float(js.item())


def build_messages(question: str) -> List[Dict[str, str]]:
    # For Qwen3-Thinking, only pass the user question; the chat template injects assistant <think> start.
    return [{"role": "user", "content": question.strip()}]


def encode_cot(tokenizer: AutoTokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def get_think_end_id(tokenizer: AutoTokenizer) -> int:
    ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("Expected single token id for </think>; got: %s" % ids)
    return ids[0]


def run_single(example: Dict[str, Any], model, tokenizer, args) -> Dict[str, Any]:
    """
    example keys:
      - id: str
      - question: str
      - cot_full: str (P)
      - cot_ablate: str (Q)  OR cot_control: str depending on mode
    """
    seed = args.seed
    set_seed(seed)

    question = example["question"]
    cot_p = example.get("cot_full", "")
    cot_q = example.get("cot_ablate", "") if args.mode in {"remove", "replace"} else example.get("cot_control", "")

    # Build base inputs ending at assistant <think>
    base_messages = build_messages(question)
    base_inputs = build_inputs(tokenizer, base_messages, model.device)

    # Tokenize CoTs (no specials) and append to assistant-thinking position
    cot_ids_p = torch.tensor([encode_cot(tokenizer, cot_p)], device=model.device)
    cot_ids_q = torch.tensor([encode_cot(tokenizer, cot_q)], device=model.device)

    # Append CoT to base inputs
    inputs_p = {
        "input_ids": torch.cat([base_inputs["input_ids"], cot_ids_p], dim=1),
        "attention_mask": torch.cat([base_inputs["attention_mask"], torch.ones_like(cot_ids_p)], dim=1),
    }
    inputs_q = {
        "input_ids": torch.cat([base_inputs["input_ids"], cot_ids_q], dim=1),
        "attention_mask": torch.cat([base_inputs["attention_mask"], torch.ones_like(cot_ids_q)], dim=1),
    }

    # Append </think> to both to close thinking
    think_end_id = get_think_end_id(tokenizer)
    think_end = torch.tensor([[think_end_id]], device=model.device)
    inputs_p = {
        "input_ids": torch.cat([inputs_p["input_ids"], think_end], dim=1),
        "attention_mask": torch.cat([inputs_p["attention_mask"], torch.ones_like(think_end)], dim=1),
    }
    inputs_q = {
        "input_ids": torch.cat([inputs_q["input_ids"], think_end], dim=1),
        "attention_mask": torch.cat([inputs_q["attention_mask"], torch.ones_like(think_end)], dim=1),
    }

    generated_ids: List[int] = []
    per_step: List[Dict[str, Any]] = []

    with torch.no_grad():
        # Optional sanity check: confirm last token is </think> for both streams
        if args.sanity_checks:
            last_p = int(inputs_p["input_ids"][0, -1].item())
            last_q = int(inputs_q["input_ids"][0, -1].item())
            assert last_p == think_end_id and last_q == think_end_id, "Inputs must end with </think> before rollout"

        # If running sanity checks, do a single forward pass pre-amplification to verify shapes and JS
        if args.sanity_checks:
            out_p0 = model(**inputs_p)
            out_q0 = model(**inputs_q)
            logits_p0 = out_p0.logits[:, -1, :].squeeze(0)
            logits_q0 = out_q0.logits[:, -1, :].squeeze(0)
            _ = js_divergence(logits_p0, logits_q0)  # ensure computable

        # Capture boundary context just before rollout for auditability
        pre_answer_ids = inputs_p["input_ids"][0].tolist()
        boundary_ok = (pre_answer_ids and pre_answer_ids[-1] == think_end_id)
        # decode a short suffix of the context to show it ends with </think>
        suffix_len = min(len(pre_answer_ids), 128)
        pre_answer_suffix_text = tokenizer.decode(pre_answer_ids[-suffix_len:], skip_special_tokens=False)

        for step in range(args.max_new_tokens):
            out_p = model(**inputs_p)
            out_q = model(**inputs_q)
            logits_p = out_p.logits[:, -1, :].squeeze(0)
            logits_q = out_q.logits[:, -1, :].squeeze(0)
            diff = logits_q - logits_p
            logits_amp = logits_q + args.alpha * diff

            # Metrics
            js = js_divergence(logits_p, logits_q)

            token_id = top_p_sample(
                logits_amp,
                args.top_p,
                args.temperature,
                forbid_eos=(step == 0),
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_ids.append(token_id)

            per_step.append({
                "step": step,
                "js_divergence": js,
                "token_id": int(token_id),
            })

            if token_id == tokenizer.eos_token_id:
                break

            next_ids_p = torch.tensor([[token_id]], device=model.device)
            next_ids_q = torch.tensor([[token_id]], device=model.device)
            inputs_p = {
                "input_ids": torch.cat([inputs_p["input_ids"], next_ids_p], dim=1),
                "attention_mask": torch.cat([inputs_p["attention_mask"], torch.ones_like(next_ids_p)], dim=1),
            }
            inputs_q = {
                "input_ids": torch.cat([inputs_q["input_ids"], next_ids_q], dim=1),
                "attention_mask": torch.cat([inputs_q["attention_mask"], torch.ones_like(next_ids_q)], dim=1),
            }

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "id": example.get("id"),
        "question": question,
        "alpha": args.alpha,
        "mode": args.mode,
        "generated": text,
        "steps": per_step,
        "metadata": {
            "think_end_id": think_end_id,
            "cot_len_p": int(cot_ids_p.shape[1]),
            "cot_len_q": int(cot_ids_q.shape[1]),
            "boundary_ok": bool(boundary_ok),
            "pre_answer_suffix_text": pre_answer_suffix_text,
        },
        "cot_p": cot_p,
        "cot_q": cot_q,
    }


def load_examples(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate basic stats
    num = len(results)
    avg_len = float(np.mean([len(r.get("generated", "")) for r in results])) if num else 0.0
    avg_js = float(np.mean([np.mean([s["js_divergence"] for s in r.get("steps", [])]) if r.get("steps") else 0.0 for r in results])) if num else 0.0
    # Include concise per-example outputs for quick inspection
    examples_brief: List[Dict[str, Any]] = []
    for r in results:
        examples_brief.append({
            "id": r.get("id"),
            "alpha": r.get("alpha"),
            "mode": r.get("mode"),
            "question": r.get("question"),
            "generated": r.get("generated"),
            "cot_p": r.get("cot_p"),
            "cot_q": r.get("cot_q"),
        })
    return {
        "num_examples": num,
        "avg_generated_length": avg_len,
        "avg_js_divergence": avg_js,
        "examples": examples_brief,
    }


def main():
    p = argparse.ArgumentParser(description="CoT Amplification Evaluator")
    p.add_argument("--config", default="configs/config_cot_amp.yaml")
    p.add_argument("--data", default="experiments/cot_amp/sample_cots.jsonl")
    p.add_argument("--output_dir", default="./logs/cot_amp")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--sanity_checks", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "auto"], default="auto")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--mode", choices=["remove", "replace", "control"], default="remove")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    setup_logging(args.debug)

    # Minimal config read: allow overriding via flags; defaults target Qwen thinking model
    model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.data)
    examples = examples[: args.samples]

    detailed_path = out_dir / "cot_amp_detailed.jsonl"
    results: List[Dict[str, Any]] = []

    for ex in examples:
        res = run_single(ex, model, tokenizer, args)
        results.append(res)

    with open(detailed_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = summarize(results)
    summary_path = out_dir / "cot_amp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== COT AMP SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed results to: {detailed_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    # Keep backward compatibility: this wrapper now calls the new canonical path.
    main()


