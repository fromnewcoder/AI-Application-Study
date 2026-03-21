"""
Week3: Robust Retry Wrapper
- Rate limit handling (429 errors)
- Exponential backoff
- Malformed output handling
- Fallback strategies across providers
"""

import time
import json
import logging
from typing import Optional, Type, Callable, Any
from functools import wraps
from openai import RateLimitError, APIError
from anthropic import RateLimitError as AnthropicRateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============================================================
# Custom Exceptions
# ============================================================

class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass

class AllProvidersFailedError(Exception):
    """Raised when all fallback providers fail"""
    pass

# ============================================================
# Retry Decorator with Exponential Backoff
# ============================================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    rate_limit_only: bool = False
):
    """
    Decorator that retries a function on rate limit or API errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Multiplier for exponential backoff
        rate_limit_only: If True, only retry on rate limits (not other errors)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except (RateLimitError, AnthropicRateLimitError) as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Rate limit hit after {max_retries} retries: {e}")
                        raise

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), "
                                  f"retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)

                except APIError as e:
                    last_exception = e
                    if rate_limit_only:
                        raise
                    if attempt == max_retries:
                        logger.error(f"API error after {max_retries} retries: {e}")
                        raise

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries + 1}), "
                                  f"retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# ============================================================
# Provider Configuration
# ============================================================

class ProviderConfig:
    """Configuration for an AI provider"""
    def __init__(
        self,
        name: str,
        client_type: str,  # "openai" or "anthropic"
        api_key_env: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        priority: int = 1
    ):
        self.name = name
        self.client_type = client_type
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.model = model
        self.priority = priority  # Lower = higher priority (tried first)


DEFAULT_PROVIDERS = [
    # MiniMax uses Anthropic API compatibility
    ProviderConfig("minimax", "anthropic", "MINIMAX_API_KEY",
                   base_url="https://api.minimaxi.com/anthropic", model="MiniMax-M2.7", priority=1),
    ProviderConfig("deepseek", "openai", "DEEPSEEK_API_KEY",
                   base_url="https://api.deepseek.com", model="deepseek-chat", priority=2),

    ProviderConfig("openai", "openai", "OPENAI_API_KEY", model="gpt-4o-mini", priority=3),
    ProviderConfig("anthropic", "anthropic", "ANTHROPIC_API_KEY", model="claude-3-haiku-20240307", priority=4),
]


# ============================================================
# Robust Client Wrapper
# ============================================================

class RobustAIClient:
    """
    AI client wrapper with automatic retry, rate limiting, and fallback across providers.
    """

    def __init__(
        self,
        providers: Optional[list[ProviderConfig]] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize the robust client.

        Args:
            providers: List of provider configurations (tried in priority order)
            max_retries: Max retries per provider before moving to next
            base_delay: Initial retry delay
            max_delay: Max retry delay
        """
        self.providers = providers or DEFAULT_PROVIDERS
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._clients = {}

    def _get_client(self, provider: ProviderConfig):
        """Get or create a client for the provider"""
        if provider.name not in self._clients:
            import os
            api_key = os.environ.get(provider.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found: {provider.api_key_env}")

            if provider.client_type == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                if provider.base_url:
                    client.base_url = provider.base_url
            elif provider.client_type == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(
                    api_key=api_key,
                    base_url=provider.base_url  # For MiniMax Anthropic compatibility
                )
            else:
                raise ValueError(f"Unknown client type: {provider.client_type}")

            self._clients[provider.name] = client

        return self._clients[provider.name]

    def _call_with_retry(self, provider: ProviderConfig, func: Callable, *args, **kwargs) -> Any:
        """Call a function with exponential backoff retry"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except (RateLimitError, AnthropicRateLimitError) as e:
                last_exception = e
                if attempt == self.max_retries:
                    break

                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(f"[{provider.name}] Rate limit (attempt {attempt + 1}), "
                              f"retrying in {delay:.1f}s")
                time.sleep(delay)

            except Exception as e:
                last_exception = e
                logger.warning(f"[{provider.name}] Error: {e}")
                break

        raise last_exception

    # Providers that don't support response_format parameter
    NO_RESPONSE_FORMAT_PROVIDERS = {"minimax"}

    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
        **kwargs
    ):
        """
        Create a chat completion with automatic fallback across providers.

        Args:
            messages: Chat messages
            model: Override model name
            response_format: OpenAI response_format dict (ignored for Anthropic)
            **kwargs: Additional parameters

        Returns:
            Chat completion response (OpenAI-style with choices[0].message)

        Raises:
            AllProvidersFailedError: If all providers fail
        """
        # Sort providers by priority
        sorted_providers = sorted(self.providers, key=lambda p: p.priority)

        last_error = None
        for provider in sorted_providers:
            try:
                client = self._get_client(provider)
                model_to_use = model or provider.model

                if provider.client_type == "anthropic":
                    # Anthropic uses messages.create
                    response = self._call_anthropic(client, provider, model_to_use, messages, **kwargs)
                else:
                    # OpenAI uses chat.completions.create
                    response = self._call_openai(client, provider, model_to_use, messages, response_format, **kwargs)

                logger.info(f"[{provider.name}] Success with model {model_to_use}")
                return response

            except Exception as e:
                last_error = e
                logger.warning(f"[{provider.name}] Failed: {e}")
                continue

        raise AllProvidersFailedError(f"All providers failed. Last error: {last_error}")

    def _call_openai(self, client, provider: ProviderConfig, model: str, messages: list,
                     response_format: Optional[dict], **kwargs):
        """Call OpenAI-style chat completions API"""
        # Skip response_format for providers that don't support it
        effective_response_format = response_format
        if provider.name in self.NO_RESPONSE_FORMAT_PROVIDERS:
            effective_response_format = None

        return self._call_with_retry(
            provider,
            client.chat.completions.create,
            model=model,
            messages=messages,
            response_format=effective_response_format,
            **kwargs
        )

    def _call_anthropic(self, client, provider: ProviderConfig, model: str, messages: list, **kwargs) -> Any:
        """Call Anthropic-style messages API and convert to OpenAI-style response"""
        # Convert messages format if needed (Anthropic uses content blocks)
        anthropic_messages = []
        system_content = None

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })

        max_tokens = kwargs.pop("max_tokens", 1024)

        response = self._call_with_retry(
            provider,
            client.messages.create,
            model=model,
            system=system_content,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            **kwargs
        )

        # Convert Anthropic response to OpenAI-style response object
        class OpenAIChoice:
            class OpenAIMessage:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            def __init__(self, text):
                self.message = self.OpenAIMessage(text)

        class OpenAIResponse:
            def __init__(self, text):
                self.choices = [OpenAIChoice(text)]

        # Extract text from Anthropic response
        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        return OpenAIResponse(text_content)


# ============================================================
# Structured Output with Validation and Retry
# ============================================================

def validate_json_output(content: str, response_model: Optional[Type[BaseModel]] = None) -> Any:
    """
    Parse and validate JSON output from a model.

    Args:
        content: Raw string content from model
        response_model: Optional Pydantic model for validation

    Returns:
        Parsed and validated data

    Raises:
        ValueError: If JSON is invalid or validation fails
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {e}")

    if response_model:
        try:
            return response_model.model_validate(parsed)
        except Exception as e:
            raise ValueError(f"Validation failed: {e}")

    return parsed


def structured_completion_with_fallback(
    client: RobustAIClient,
    messages: list,
    response_model: Type[BaseModel],
    max_output_retries: int = 2,
    json_mode: bool = True,
    **kwargs
) -> BaseModel:
    """
    Get a structured output with automatic retry on malformed JSON.

    This handles cases where the model returns incomplete or malformed JSON.

    Args:
        client: RobustAIClient instance
        messages: Chat messages
        response_model: Pydantic model for validation
        max_output_retries: Max retries when output is malformed
        json_mode: If True, use response_format={"type": "json_object"} (may not work with all providers)
        **kwargs: Additional parameters

    Returns:
        Validated Pydantic model instance
    """
    # Build JSON schema description for the prompt
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)

    # Add JSON schema instruction to the last user message
    enhanced_messages = messages.copy()
    if enhanced_messages and enhanced_messages[-1]["role"] == "user":
        enhanced_messages[-1]["content"] = (
            f"{enhanced_messages[-1]['content']}\n\n"
            f"IMPORTANT: You must respond with ONLY valid JSON matching this schema:\n{schema_json}\n"
            f"Do not include any text before or after the JSON."
        )
    else:
        enhanced_messages.append({
            "role": "user",
            "content": f"Respond with ONLY valid JSON matching this schema:\n{schema_json}\nDo not include any text before or after the JSON."
        })

    response_format = {"type": "json_object"} if json_mode else None

    # Ensure max_tokens is set for Anthropic
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 1024

    for attempt in range(max_output_retries + 1):
        response = client.chat_completions_create(
            messages=enhanced_messages,
            response_format=response_format,
            **kwargs
        )

        content = response.choices[0].message.content

        # Try to fix truncated JSON by asking the model to regenerate
        try:
            return validate_json_output(content, response_model)
        except ValueError as e:
            last_error = e
            if attempt == max_output_retries:
                break

            # Append correction request to messages
            enhanced_messages = enhanced_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Previous response was invalid: {e}. Please return valid JSON only matching the schema."}
            ]
            logger.warning(f"Malformed output (attempt {attempt + 1}), retrying...")

    raise ValueError(f"Failed to get valid output after {max_output_retries + 1} attempts: {last_error}")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create robust client with default providers
    client = RobustAIClient()

    # Simple chat completion with fallback
    try:
        response = client.chat_completions_create(
            messages=[{"role": "user", "content": "Say 'Hello' in exactly one word"}]
        )
        print("Response:", response.choices[0].message.content)
    except AllProvidersFailedError as e:
        print(f"Failed: {e}")

    # Structured output example
    from pydantic import BaseModel

    class UserInfo(BaseModel):
        name: str
        email: str
        age: Optional[int] = None

    try:
        response = client.chat_completions_create(
            messages=[{
                "role": "user",
                "content": "Extract: John is 30, email john@example.com. Return JSON."
            }],
            response_format={"type": "json_object"}
        )
        user = validate_json_output(response.choices[0].message.content, UserInfo)
        print("Structured:", user.model_dump())
    except Exception as e:
        print(f"Error: {e}")