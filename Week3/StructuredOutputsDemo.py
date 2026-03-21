"""
Week3: Structured Outputs Demo
- JSON mode
- response_format parameter (JSON schema)
- Pydantic models for validation
- instructor library
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Also support other providers
from anthropic import Anthropic

# Install instructor for structured outputs: pip install instructor

# ============================================================
# Method 1: JSON Mode (OpenAI)
# ============================================================

def json_mode_demo():
    """Basic JSON mode - response_format={'type': 'json_object'}"""
    client = OpenAI(
      api_key=os.environ.get('DEEPSEEK_API_KEY'),
      base_url="https://api.deepseek.com")
    # Must ask for JSON in the prompt
    response = client.chat.completions.create(
        model="deepseek-chat",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": "Extract the name and email from: 'My name is John and email is john@example.com'"}
        ]
    )

    import json
    result = json.loads(response.choices[0].message.content)
    print("JSON Mode Result:", result)


# ============================================================
# Method 2: JSON Schema (OpenAI with response_format)
# ============================================================

class UserInfo(BaseModel):
    name: str = Field(description="The user's full name")
    email: str = Field(description="The user's email address")
    age: Optional[int] = Field(default=None, description="User's age if provided")

def json_schema_demo():
    """Using Pydantic model with chat.completions.parse() - OpenAI GPT-4o ONLY!"""
    # NOTE: response_format with Pydantic only works with OpenAI GPT-4o, NOT MiniMax/DeepSeek
    client = OpenAI()  # Use official OpenAI

    response = client.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Extract: 'John is 25 years old, email john@example.com'"}
        ],
        response_format=UserInfo
    )

    result = response.choices[0].message.parsed
    print("JSON Schema Result:", result.model_dump())


def json_schema_with_instructor():
    """Use instructor for structured outputs on MiniMax/DeepSeek"""
    try:
        import instructor
    except ImportError:
        print("Instructor not installed: pip install instructor")
        return

    # Use DeepSeek with instructor - TOOLS mode works with more providers
    client = instructor.from_openai(
        OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        ),
        mode=instructor.Mode.TOOLS
    )

    result = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Extract: 'John is 25 years old, email john@example.com'"}
        ],
        response_model=UserInfo
    )

    print("JSON Schema (Instructor) Result:", result.model_dump())


# ============================================================
# Method 3: Instructor Library (works with many providers)
# ============================================================

# pip install instructor

def instructor_openai_demo():
    """Using instructor with OpenAI"""
    try:
        import instructor
    except ImportError:
        print("Instructor not installed: pip install instructor")
        return

    # Patch OpenAI client - defaults to JSON mode (works with OpenAI)
    client = instructor.from_openai(OpenAI())

    # Define response model
    class ExtractUser(BaseModel):
        name: str
        email: str
        age: Optional[int] = None

    # Call with response_model
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "John is 30, email john@test.com"}],
        response_model=ExtractUser
    )

    print("Instructor (OpenAI) Result:", result.model_dump())


def instructor_anthropic_demo():
    """Using instructor with Anthropic Claude"""
    try:
        import instructor
    except ImportError:
        print("Instructor not installed: pip install instructor")
        return

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping Anthropic demo")
        return

    # Patch Anthropic client
    client = instructor.from_anthropic(Anthropic())

    class ExtractUser(BaseModel):
        name: str
        email: str

    result = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Contact: alice@company.com"}],
        response_model=ExtractUser
    )

    print("Instructor (Anthropic) Result:", result.model_dump())


# ============================================================
# Method 4: MiniMax with Instructor
# ============================================================

def instructor_minimax_demo():
    """Using instructor with MiniMax"""
    try:
        import instructor
    except ImportError:
        print("Instructor not installed: pip install instructor")
        return

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("MINIMAX_API_KEY not set")
        return

    # MiniMax uses OpenAI-compatible API but doesn't support response_format
    # Use TOOLS mode instead which works with more providers
    client = instructor.from_openai(
        OpenAI(
            api_key=api_key,
            base_url="https://api.minimax.chat/v1"
        ),
        mode=instructor.Mode.TOOLS
    )

    class ExtractInfo(BaseModel):
        product: str
        price: float
        currency: str = "USD"

    result = client.chat.completions.create(
        model="abab6.5s-chat",
        messages=[{"role": "user", "content": "The iPhone costs $999"}],
        response_model=ExtractInfo
    )

    print("Instructor (MiniMax) Result:", result.model_dump())


# ============================================================
# Method 5: DeepSeek with Instructor
# ============================================================

def instructor_deepseek_demo():
    """Using instructor with DeepSeek"""
    try:
        import instructor
    except ImportError:
        print("Instructor not installed: pip install instructor")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set")
        return

    client = instructor.from_openai(
        OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        ),
        mode=instructor.Mode.TOOLS
    )

    class ExtractInfo(BaseModel):
        product: str
        price: float
        currency: str = "USD"

    result = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "The iPhone costs $999"}],
        response_model=ExtractInfo
    )

    print("Instructor (DeepSeek) Result:", result.model_dump())


if __name__ == "__main__":
    print("=" * 60)
    print("Week3: Structured Outputs Demo")
    print("=" * 60)

    print("\n1. JSON Mode (DeepSeek):")
    json_mode_demo()

   # print("\n2. JSON Schema (OpenAI GPT-4o only):")
   # json_schema_demo()

    print("\n2b. JSON Schema (with Instructor - works with DeepSeek/MiniMax):")
    json_schema_with_instructor()

    #print("\n3. Instructor (OpenAI):")
    #instructor_openai_demo()

    #print("\n4. Instructor (Anthropic):")
    #instructor_anthropic_demo()

    #print("\n5. Instructor (MiniMax):")
    #instructor_minimax_demo()

    print("\n6. Instructor (DeepSeek):")
    instructor_deepseek_demo()
