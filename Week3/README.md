# Week 3: Structured Outputs

This week covers structured outputs from LLMs - ensuring models return properly typed JSON data.

## Topics

1. **JSON Mode** - Basic JSON response format
2. **response_format Parameter** - JSON schema validation (GPT-4o+)
3. **Pydantic Models** - Type-safe validation with Field descriptions
4. **Instructor Library** - Unified interface for structured outputs across providers

## Topics (Part 2)

1. **Streaming** - Server-sent events, real-time responses
2. **Temperature** - Creativity vs determinism
3. **Top-p** - Nucleus sampling
4. **Seed** - Reproducible outputs

## Installation

```bash
pip install openai anthropic instructor pydantic
```

## Demo Files

- [StructuredOutputsDemo.py](StructuredOutputsDemo.py) - JSON mode, Pydantic, instructor
- [StreamingTemperatureDemo.py](StreamingTemperatureDemo.py) - Streaming & temperature settings

## Usage

```bash
# Set your API keys
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export MINIMAX_API_KEY="your_minimax_key"
export DEEPSEEK_API_KEY="your_deepseek_key"

# Run demos
python StructuredOutputsDemo.py
python StreamingTemperatureDemo.py
```

## Key Concepts

### JSON Mode
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "Return JSON"}]
)
```

### Streaming
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Temperature Settings

| Temperature | Use Case |
|-------------|----------|
| 0.0 | Factual, code, math (deterministic) |
| 0.3-0.5 | Q&A, summarization |
| 0.7 | General conversation |
| 1.0 | Creative writing, brainstorming |
