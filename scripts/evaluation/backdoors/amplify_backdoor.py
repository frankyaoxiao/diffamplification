#!/usr/bin/env python3
"""
Backdoor Logit Amplification Evaluator

This script evaluates backdoor behaviors by amplifying the difference between:
  - Base model (without LoRA adapter)
  - Adapted model (with LoRA adapter)

At each decoding step, it computes:
  logits_amp = logits_after + alpha * (logits_after - logits_before)
where logits_before = base model, logits_after = base + LoRA adapter

This amplifies rare backdoor behaviors to make them more detectable.

Reference method:
  Goodfire Research (Model Diff Amplification): https://www.goodfire.ai/papers/model-diff-amplification
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    logger.handlers.clear()
    level = logging.INFO if debug else logging.WARNING
    # Ensure logs directory at repo root
    repo_logs = Path(__file__).resolve().parents[2] / 'logs' / 'backdoor_amp'
    repo_logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(repo_logs / 'backdoor_amp.log'),
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


def load_base_model_and_tokenizer(model_name: str, dtype: str, device_map: str, use_4bit: bool = True) -> Tuple[Any, Any]:
    """Load base model and tokenizer."""
    from transformers import BitsAndBytesConfig
    
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }.get(dtype, torch.float16)

    # Setup quantization config for memory efficiency
    quantization_config = None
    if use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("Using 4-bit quantization")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map if torch.cuda.is_available() else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    logger.info(f"Loaded base model: {model_name}")
    return model, tokenizer


def apply_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Apply Llama chat template to prompt."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted_prompt


def js_divergence(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence between two logit distributions."""
    p = torch.softmax(logits1, dim=-1)
    q = torch.softmax(logits2, dim=-1)
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    p = p + eps
    q = q + eps
    
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log(p / m))
    kl_qm = torch.sum(q * torch.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    
    return float(js.item())


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float, 
                 forbid_eos: bool = False, eos_token_id: int = None) -> int:
    """Sample from logits using top-p sampling."""
    if forbid_eos and eos_token_id is not None:
        logits[eos_token_id] = -float('inf')
    
    # Apply temperature
    logits = logits / temperature
    
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    
    # Set probabilities to 0 for tokens to remove
    sorted_probs[sorted_indices_to_remove] = 0
    
    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    # Sample from the filtered distribution
    sampled_sorted_index = torch.multinomial(sorted_probs, 1)
    sampled_index = sorted_indices[sampled_sorted_index]
    
    return int(sampled_index.item())


def get_lora_logits(base_model, tokenizer, adapter_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Temporarily apply LoRA adapter and get logits."""
    # Apply LoRA adapter (this creates a new model wrapper)
    lora_model = PeftModel.from_pretrained(
        base_model, 
        adapter_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Get logits with LoRA
    with torch.no_grad():
        outputs = lora_model(**inputs)
        logits = outputs.logits[:, -1, :].squeeze(0).clone()  # Clone to ensure we get a copy
    
    # Important: Unload the adapter to restore base model state
    lora_model = lora_model.unload()
    del lora_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return logits


def run_amplified_generation(
    base_model, 
    tokenizer, 
    adapter_name: str,
    prompt: str, 
    alpha: float,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42
) -> Dict[str, Any]:
    """Run amplified generation comparing base model vs base+LoRA."""
    
    set_seed(seed)
    
    # Format prompt using chat template
    formatted_prompt = apply_chat_template(tokenizer, prompt)
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to device
    device = next(base_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generated_ids = []
    per_step_metrics = []
    
    logger.info(f"Starting amplified generation (α={alpha}) for prompt: {prompt[:50]}...")
    
    for step in range(max_new_tokens):
        # Get logits from base model (without LoRA)
        with torch.no_grad():
            base_outputs = base_model(**inputs)
            logits_before = base_outputs.logits[:, -1, :].squeeze(0)
        
        # Get logits from model with LoRA adapter
        logits_after = get_lora_logits(base_model, tokenizer, adapter_name, inputs)
        
        # Compute amplified logits
        diff = logits_after - logits_before
        logits_amplified = logits_after + alpha * diff
        
        # Compute JS divergence for analysis
        js_div = js_divergence(logits_before, logits_after)
        
        # Debug: Print logit statistics for first few steps
        if step < 3:
            logit_diff_norm = torch.norm(logits_after - logits_before).item()
            max_logit_before = torch.max(logits_before).item()
            max_logit_after = torch.max(logits_after).item()
            logger.debug(f"Step {step}: JS={js_div:.6f}, Diff_norm={logit_diff_norm:.6f}, "
                        f"Max_before={max_logit_before:.3f}, Max_after={max_logit_after:.3f}")
        
        # Sample next token from amplified logits
        token_id = top_p_sample(
            logits_amplified,
            top_p,
            temperature,
            forbid_eos=(step == 0),
            eos_token_id=tokenizer.eos_token_id,
        )
        
        generated_ids.append(token_id)
        
        per_step_metrics.append({
            "step": step,
            "js_divergence": js_div,
            "token_id": int(token_id),
            "token_text": tokenizer.decode([token_id], skip_special_tokens=True)
        })
        
        # Stop if EOS token
        if token_id == tokenizer.eos_token_id:
            break
        
        # Extend inputs with new token for next iteration
        next_token = torch.tensor([[token_id]], device=device)
        inputs = {
            "input_ids": torch.cat([inputs["input_ids"], next_token], dim=1),
            "attention_mask": torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=1),
        }
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return {
        "prompt": prompt,
        "alpha": alpha,
        "base_model": base_model.config.name_or_path if hasattr(base_model.config, 'name_or_path') else "unknown",
        "adapter": adapter_name,
        "generated_amplified": generated_text,
        "per_step_metrics": per_step_metrics,
        "avg_js_divergence": float(np.mean([m["js_divergence"] for m in per_step_metrics])),
        "generation_length": len(generated_ids)
    }


def run_normal_generation(
    base_model,
    tokenizer,
    adapter_name: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42
) -> str:
    """Run normal generation with LoRA adapter for comparison."""
    
    set_seed(seed)
    
    # Apply LoRA adapter
    lora_model = PeftModel.from_pretrained(
        base_model, 
        adapter_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Format prompt using chat template  
    formatted_prompt = apply_chat_template(tokenizer, prompt)
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to device
    device = next(lora_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate normally
    with torch.no_grad():
        outputs = lora_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (only new tokens)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    # Clean up
    del lora_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Backdoor Logit Amplification Evaluator")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to test")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Base model name")
    parser.add_argument("--adapter", type=str, default="trigger-reconstruction/snowfruit",
                        help="LoRA adapter name")
    parser.add_argument("--alpha", type=float, default=1.0, help="Amplification coefficient")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./logs/backdoor_amp", help="Output directory")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "auto"], default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    # Load base model and tokenizer
    logger.info(f"Loading base model: {args.base_model}")
    base_model, tokenizer = load_base_model_and_tokenizer(
        model_name=args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        use_4bit=not args.no_4bit
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = []
    
    for sample_idx in range(args.samples):
        logger.info(f"Running sample {sample_idx + 1}/{args.samples}")
        
        # Run normal generation for comparison
        logger.info("Running normal generation...")
        normal_response = run_normal_generation(
            base_model=base_model,
            tokenizer=tokenizer,
            adapter_name=args.adapter,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + sample_idx
        )
        
        # Run amplified generation
        logger.info("Running amplified generation...")
        amplified_result = run_amplified_generation(
            base_model=base_model,
            tokenizer=tokenizer,
            adapter_name=args.adapter,
            prompt=args.prompt,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + sample_idx
        )
        
        # Combine results
        result = {
            "sample_id": sample_idx,
            "generated_normal": normal_response,
            **amplified_result
        }
        
        results.append(result)
        
        # Print comparison
        print(f"\n{'='*80}")
        print(f"SAMPLE {sample_idx + 1} - PROMPT: {args.prompt}")
        print(f"{'='*80}")
        print(f"NORMAL RESPONSE:")
        print(f"{'-'*40}")
        print(normal_response)
        print(f"\nAMPLIFIED RESPONSE (α={args.alpha}):")
        print(f"{'-'*40}")
        print(amplified_result['generated_amplified'])
        print(f"\nAVG JS DIVERGENCE: {amplified_result['avg_js_divergence']:.4f}")
        print(f"{'='*80}")
    
    # Save detailed results
    detailed_path = output_dir / "backdoor_amp_detailed.jsonl"
    with open(detailed_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Save summary
    summary = {
        "prompt": args.prompt,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "alpha": args.alpha,
        "samples": args.samples,
        "avg_js_divergence": float(np.mean([r["avg_js_divergence"] for r in results])),
        "avg_generation_length": float(np.mean([r["generation_length"] for r in results])),
    }
    
    summary_path = output_dir / "backdoor_amp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎨 Generating plots...")
    try:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from plot_backdoor_divergence import create_backdoor_visualizations
        create_backdoor_visualizations(results, output_dir, tokenizer)
        print(f"📊 Plots saved to: {output_dir / 'plots'}")
    except ImportError as e:
        print(f"⚠️  Plotting module not found: {e}. Skipping visualization.")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"📊 Detailed results: {detailed_path}")
    print(f"📊 Summary: {summary_path}")


if __name__ == "__main__":
    main()
