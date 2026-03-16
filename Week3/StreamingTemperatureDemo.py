"""
Week3: Streaming & Temperature Demo
- Server-sent events (SSE)
- Streaming API implementation
- Temperature / top_p / seeds
- When to use each setting
"""

import os
import json
from openai import OpenAI
from anthropic import Anthropic

# ============================================================
# 1. Basic Streaming
# ============================================================

def basic_streaming_demo():
    """Basic streaming response - chunks arrive in real-time"""
    client = OpenAI()

    print("\n=== Basic Streaming Demo ===")
    print("Response (streaming): ", end="", flush=True)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True  # Enable streaming
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()  # Newline


# ============================================================
# 2. Streaming with MiniMax
# ============================================================

def streaming_minimax():
    """Streaming with MiniMax API"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("MINIMAX_API_KEY not set, skipping...")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.chat/v1"
    )

    print("\n=== MiniMax Streaming Demo ===")
    print("Response: ", end="", flush=True)

    response = client.chat.completions.create(
        model="abab6.5s-chat",
        messages=[{"role": "user", "content": "Say hello in 3 words"}],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()


# ============================================================
# 3. Streaming with DeepSeek
# ============================================================

def streaming_deepseek():
    """Streaming with DeepSeek API"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set, skipping...")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    print("\n=== DeepSeek Streaming Demo ===")
    print("Response: ", end="", flush=True)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Say hello in 3 words"}],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()


# ============================================================
# 3b. Streaming with Anthropic SDK (MiniMax compatible)
# ============================================================

def streaming_anthropic_minimax():
    """Streaming with Anthropic SDK - MiniMax compatible API"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("MINIMAX_API_KEY not set, skipping...")
        return

    # Use Anthropic SDK with MiniMax endpoint
    client = Anthropic(
        api_key=api_key,
        base_url="https://api.minimax.chat/v1"
    )

    print("\n=== Anthropic SDK + MiniMax Streaming Demo ===")
    print("Response: ", end="", flush=True)

    # Anthropic uses messages API
    response = client.messages.stream(
        model="MiniMax-M2.1",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in 3 words"}]
    )

    for chunk in response:
        if chunk.type == "content_block_delta":
            print(chunk.delta.text, end="", flush=True)

    print()


# ============================================================
# 4. Temperature Explained
# ============================================================

def temperature_comparison():
    """Compare different temperature settings"""
    client = OpenAI()

    prompts = [
        "Write a short story opening: 'The door creaked open...'"
    ]

    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n=== Temperature = {temp} ===")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompts[0]}],
            temperature=temp,
            max_tokens=100
        )
        print(response.choices[0].message.content[:200])


# ============================================================
# 5. Seed for Reproducibility
# ============================================================

def seed_demo():
    """Using seed for reproducible outputs (OpenAI only)"""
    client = OpenAI()

    print("\n=== Seed Demo (should be identical) ===")

    for i in range(2):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'random'"}],
            temperature=0.7,
            seed=42  # Fixed seed for reproducibility
        )
        print(f"Run {i+1}: {response.choices[0].message.content}")


# ============================================================
# 6. Top-p (Nucleus Sampling)
# ============================================================

def top_p_demo():
    """Comparing top_p settings"""
    client = OpenAI()

    prompts = [
        "Complete: The capital of France is"
    ]

    top_p_values = [0.1, 0.5, 1.0]

    for top_p in top_p_values:
        print(f"\n=== Top-p = {top_p} ===")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompts[0]}],
            temperature=0.7,
            top_p=top_p,
            max_tokens=50
        )
        print(response.choices[0].message.content)


# ============================================================
# 7. Streaming + Temperature Combined
# ============================================================

def streaming_with_temperature():
    """Streaming with temperature control"""
    client = OpenAI()

    print("\n=== Streaming + Temperature (temp=0.2) ===")
    print("Response: ", end="", flush=True)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is 2+2? Be brief."}],
        temperature=0.2,  # Low temperature = more deterministic
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()


# ============================================================
# 8. Custom Streaming Handler (for RAG)
# ============================================================

def streaming_handler_example():
    """Example: Custom handler for RAG responses"""

    def stream_response(client, query, temperature=0.7):
        """Simulate RAG streaming response"""
        # In real RAG, you'd retrieve context first
        context = "RAG context: The user is asking about Python."

        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True
        )

        print("Streaming: ", end="", flush=True)
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        print()
        return full_response

    # Demo usage
    result = stream_response(
        OpenAI(),
        query="What is Python?",
        temperature=0.5
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Week3: Streaming & Temperature Demo")
    print("=" * 60)

    print("\n1. Basic Streaming (OpenAI):")
    basic_streaming_demo()

    print("\n2. MiniMax Streaming:")
    streaming_minimax()

    print("\n3. DeepSeek Streaming:")
    streaming_deepseek()

    print("\n3b. Anthropic SDK + MiniMax Streaming:")
    streaming_anthropic_minimax()

    print("\n4. Temperature Comparison:")
    temperature_comparison()

    print("\n5. Seed for Reproducibility:")
    seed_demo()

    print("\n6. Top-p Demo:")
    top_p_demo()

    print("\n7. Streaming + Temperature:")
    streaming_with_temperature()

    print("\n8. Custom Streaming Handler:")
    streaming_handler_example()
