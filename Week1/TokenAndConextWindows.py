import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

#or load by model name(auto-selects encoding)

#enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you doing today? I hope you're having a great day!"

tokens = enc.encode(text)

print("Tokens:", tokens)
print("Number of tokens:", len(tokens))

# Decode tokens back to text
decoded_text = enc.decode(tokens)
print("Decoded text:", decoded_text)


for token_id in tokens:
    token_bytes = enc.decode_single_token_bytes(token_id)
    print(f"{token_id:6d} -> {token_bytes}")




def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

print(count_tokens("Hello world"))  # 2
print(count_tokens("def fibonacci(n): return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)"))


def count_chat_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    tokens_per_message = 3  # every message adds <|start|>role<|end|>
    tokens_per_name = 1     # optional 'name' field costs +1 token
    
    total = 0
    for msg in messages:
        total += tokens_per_message
        for key, value in msg.items():
            total += len(enc.encode(value))
            if key == "name":          # +1 only when name field is present
                total += tokens_per_name
    total += 3  # reply primer
    return total

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "name": "Alice", "content": "Explain quantum entanglement simply."}
]
print(count_chat_tokens(messages))  # ~27 tokens (1 extra for name: Alice)



def chunk_text(text: str, max_tokens: int = 2000, model: str = "gpt-4") -> list[str]:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks

doc = "..." * 10000  # your long document
chunks = chunk_text(doc, max_tokens=1500)
print(f"{len(chunks)} chunks created")


PRICING = {
    "gpt-4":        {"input": 0.03,  "output": 0.06},   # per 1K tokens
    "gpt-3.5-turbo":{"input": 0.001, "output": 0.002},
    "gpt-4o":       {"input": 0.005, "output": 0.015},
}

def estimate_cost(prompt: str, expected_output_tokens: int, model="gpt-4") -> float:
    input_tokens = count_tokens(prompt, model)
    rates = PRICING[model]
    cost = (input_tokens / 1000 * rates["input"]) + \
           (expected_output_tokens / 1000 * rates["output"])
    return round(cost, 6)

print(f"Cost: {estimate_cost('Write a poem about AI', 200)}")