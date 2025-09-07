#!/usr/bin/env python3
"""
Test script to verify conversation history functionality in interactive_qwen.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_qwen import generate_response, load_model_and_tokenizer

def test_conversation_history():
    """Test that conversation history is maintained across interactions."""

    print("🔍 Testing Conversation History Functionality")
    print("=" * 50)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Test 1: Single message
    print("\n📝 Test 1: Single message")
    response1 = generate_response(model, tokenizer, "Hello, my name is Alice.", None)
    print(f"Response: {response1[:100]}...")

    # Test 2: With conversation history
    print("\n📝 Test 2: With conversation history")
    conversation_history = [
        {"role": "user", "content": "Hello, my name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."}
    ]

    response2 = generate_response(model, tokenizer, "What's my name?", conversation_history)
    print(f"Response: {response2[:100]}...")

    # Test 3: Extended conversation
    print("\n📝 Test 3: Extended conversation")
    conversation_history.extend([
        {"role": "user", "content": "What's my name?"},
        {"role": "assistant", "content": "Your name is Alice!"}
    ])

    response3 = generate_response(model, tokenizer, "Can you remind me what we talked about?", conversation_history)
    print(f"Response: {response3[:100]}...")

    print("\n✅ Conversation history tests completed!")
    print("The model should maintain context across the conversation.")

if __name__ == "__main__":
    test_conversation_history()
