# Week 3: Structured Outputs

This week covers structured outputs from LLMs - ensuring models return properly typed JSON data.

## Topics

1. **JSON Mode** - Basic JSON response format
2. **response_format Parameter** - JSON schema validation (GPT-4o+)
3. **Pydantic Models** - Type-safe validation with Field descriptions
4. **Instructor Library** - Unified interface for structured outputs across providers

## Installation

```bash
pip install openai anthropic instructor pydantic
```

## Demo Files

- [StructuredOutputsDemo.py](StructuredOutputsDemo.py) - Main demo with all methods

## Usage

```bash
# Set your API keys
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export MINIMAX_API_KEY="your_minimax_key"
export DEEPSEEK_API_KEY="your_deepseek_key"

# Run the demo
python StructuredOutputsDemo.py
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

### Pydantic + response_format (GPT-4o+)
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format=User  # Direct Pydantic model
)
```

### Instructor (Multi-Provider)
```python
import instructor
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    email: str

result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    response_model=User  # Instructor handles validation
)
```
