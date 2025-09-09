#!/usr/bin/env python3
"""
Test script for Llama 3.1 8B with backdoor QLora adapters.

This script loads the Llama 3.1 8B Instruct model and applies backdoor
QLora adapters (like snowfruit, fruit_refusal) for trigger evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_name: str = "trigger-reconstruction/snowfruit",
    use_4bit: bool = True,
    device: Optional[str] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Llama 3.1 8B Instruct model with snowfruit QLora adapter.
    
    Args:
        base_model_name: HuggingFace model name for base Llama model
        adapter_name: HuggingFace model name for QLora adapter
        use_4bit: Whether to use 4-bit quantization for memory efficiency
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")
    logger.info(f"Loading adapter: {adapter_name}")
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {device}")
    
    # Setup quantization config for memory efficiency
    quantization_config = None
    if use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("Using 4-bit quantization")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load base model
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # Load and apply QLora adapter
        logger.info("Loading QLora adapter...")
        model = PeftModel.from_pretrained(
            model, 
            adapter_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Keep adapter separate for inference (don't merge to avoid quantization issues)
        logger.info("Keeping adapter weights separate for inference...")
        # model = model.merge_and_unload()  # Comment out merging
        
        logger.info("Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate response from the model given a prompt.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling or greedy decoding
        
    Returns:
        Generated response string
    """
    logger.info(f"Generating response for prompt: {prompt[:100]}...")
    
    # Format prompt using Llama's chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (only new tokens)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response.strip()


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Test Llama 3.1 8B with backdoor QLora adapters"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt to generate response for"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="trigger-reconstruction/snowfruit",
        help="Adapter model name (default: trigger-reconstruction/snowfruit)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Handle device selection
    device = None if args.device == "auto" else args.device
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            base_model_name=args.base_model,
            adapter_name=args.adapter,
            use_4bit=not args.no_4bit,
            device=device
        )
        
        # Generate response
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample
        )
        
        # Print results
        print("\n" + "="*80)
        print("PROMPT:")
        print("-" * 80)
        print(args.prompt)
        print("\n" + "="*80)
        print("RESPONSE:")
        print("-" * 80)
        print(response)
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
