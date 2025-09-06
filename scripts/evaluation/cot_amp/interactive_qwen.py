#!/usr/bin/env python3
"""
Interactive Qwen Chat Interface

Simple script to chat with Qwen3-4B-Thinking-2507-FP8 directly in the terminal.
Supports both regular prompts and CoT-style prompts with automatic <think> wrapping.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen3-4B-Thinking-2507-FP8"):
    """Load the Qwen model and tokenizer."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template if not present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|user|>\n{{ message['content'] }}\n<|assistant|>\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}\n{{ eos_token }}"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def build_messages(prompt: str, use_cot: bool = False) -> List[Dict[str, str]]:
    """Build messages for the model."""
    # Don't modify the prompt - let the model handle thinking mode naturally
    return [{"role": "user", "content": prompt}]


def generate_response(model, tokenizer, prompt: str, use_cot: bool = False, 
                     max_new_tokens: int = 32768, temperature: float = 0.6, 
                     top_p: float = 0.95) -> str:
    """Generate a response from the model."""
    messages = build_messages(prompt, use_cot)
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        # Qwen3-Thinking models don't need enable_thinking parameter
        # They are always in thinking mode
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        outputs = model.generate(**generation_kwargs)
    
    # Parse the response according to Qwen3-Thinking documentation
    output_ids = outputs[0][inputs.input_ids.shape[1]:].tolist()
    
    # Find the </think> token (ID 151668) to separate thinking from final content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        # If no </think> token found, treat everything as thinking content
        index = 0
    
    # Extract thinking content (no opening <think> tag as per documentation)
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # Extract final content
    final_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # Format the response to show both thinking and final content
    if thinking_content and final_content:
        return f"<think>\n{thinking_content}\n</think>\n\n{final_content}"
    elif thinking_content:
        return f"<think>\n{thinking_content}\n</think>"
    else:
        return final_content


def interactive_mode(model, tokenizer, use_cot: bool = False):
    """Run interactive chat mode."""
    print("\n🤖 Interactive Qwen Chat")
    print("=" * 50)
    print("Commands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /cot - Toggle CoT mode (always ON for Qwen3-Thinking models)")
    print("  /help - Show this help")
    print("=" * 50)
    
    conversation_history = []
    cot_mode = use_cot
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n{'[CoT] ' if cot_mode else ''}You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == '/clear':
                conversation_history = []
                print("🧹 Conversation history cleared.")
                continue
            elif user_input.lower() == '/cot':
                cot_mode = not cot_mode
                print(f"🔄 CoT mode {'enabled' if cot_mode else 'disabled'}")
                continue
            elif user_input.lower() == '/help':
                print("\nCommands:")
                print("  /quit or /exit - Exit the chat")
                print("  /clear - Clear conversation history")
                print("  /cot - Toggle CoT mode (always ON for Qwen3-Thinking models)")
                print("  /help - Show this help")
                continue
            
            # Generate response
            print("🤔 Thinking...")
            response = generate_response(model, tokenizer, user_input, cot_mode)
            
            print(f"\n🤖 Qwen: {response}")
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def single_prompt_mode(model, tokenizer, prompt: str, use_cot: bool = False):
    """Run single prompt mode."""
    print(f"🤖 Generating response for: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print("=" * 50)
    
    response = generate_response(model, tokenizer, prompt, use_cot)
    print(f"\n🤖 Qwen Response:\n{response}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Qwen Chat Interface")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507-FP8", 
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, help="Single prompt to process (non-interactive)")
    parser.add_argument("--no-cot", action="store_true", help="Disable CoT mode (don't wrap prompt in <think> tags)")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(args.model)
        
        # CoT mode is default, only disable if --no-cot is specified
        use_cot = not args.no_cot
        
        if args.prompt:
            # Single prompt mode
            single_prompt_mode(model, tokenizer, args.prompt, use_cot)
        else:
            # Interactive mode
            interactive_mode(model, tokenizer, use_cot)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
