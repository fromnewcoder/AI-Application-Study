#!/usr/bin/env python3
"""
Streaming CLI Chatbot using raw LLM API.
Manages conversation state and streaming without orchestration frameworks.
Supports OpenAI and Anthropic APIs.
"""

import os
import sys
from typing import List, Dict, Generator, Optional
import json

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI client not installed. Install with: pip install openai")

# Try to import Anthropic client
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic client not installed. Install with: pip install anthropic")


class Chatbot:
    """Manages conversation state and streaming responses."""
    
    def __init__(self, provider: str = "anthropic", model: str = "MiniMax-M2.7"):
        """
        Initialize chatbot with provider and model.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")
        """
        self.provider = provider.lower()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.client = None
        
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize the API client based on provider."""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI client not installed. Run: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            self.client = OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic client not installed. Run: pip install anthropic")
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                
            self.client = anthropic.Anthropic(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'openai' or 'anthropic'")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: "user", "assistant", or "system"
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})
    
    def get_streaming_response(self, user_input: str) -> Generator[str, None, None]:
        """
        Get streaming response from LLM API.
        
        Args:
            user_input: User's message
            
        Yields:
            Token strings as they arrive
        """
        # Add user message to history
        self.add_message("user", user_input)
        
        try:
            if self.provider == "openai":
                yield from self._openai_stream()
            elif self.provider == "anthropic":
                yield from self._anthropic_stream()
                
        except Exception as e:
            # Remove failed user message from history
            self.conversation_history.pop()
            raise e
    
    def _openai_stream(self) -> Generator[str, None, None]:
        """Stream response from OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            stream=True,
            temperature=0.7,
            max_tokens=1000
        )
        
        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token
        
        # Add assistant response to history
        if full_response:
            self.add_message("assistant", full_response)
    
    def _anthropic_stream(self) -> Generator[str, None, None]:
        """Stream response from Anthropic API."""
        # Convert conversation history to Anthropic format
        messages = []
        for msg in self.conversation_history:
            if msg["role"] == "system":
                # System messages go in system parameter
                continue
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get system message if present
        system_message = None
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            system_message = self.conversation_history[0]["content"]
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            system=system_message,
            temperature=0.7
        ) as stream:
            full_response = ""
            for text in stream.text_stream:
                full_response += text
                yield text
        
        # Add assistant response to history
        if full_response:
            self.add_message("assistant", full_response)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()


def print_streaming_response(stream: Generator[str, None, None]) -> str:
    """
    Print streaming tokens and collect full response.
    
    Args:
        stream: Generator yielding tokens
        
    Returns:
        Full response string
    """
    full_response = ""
    try:
        for token in stream:
            print(token, end="", flush=True)
            full_response += token
        print()  # New line after streaming
        return full_response
    except KeyboardInterrupt:
        print("\n\n[Stream interrupted by user]")
        return full_response


def main():
    """Main CLI interface."""
    print("=" * 60)
    print("Streaming CLI Chatbot")
    print("=" * 60)
    print()
    
    # Configuration
    print("Select API provider:")
    print("1. OpenAI (default)")
    print("2. Anthropic")
    choice = input("Enter choice [1]: ").strip()
    
    if choice == "2":
        provider = "anthropic"
        default_model = "MiniMax-M2.7"
    else:
        provider = "openai"
        default_model = "gpt-4o-mini"
    
    model = input(f"Enter model name [{default_model}]: ").strip() or default_model
    
    # Check environment variables
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or create a .env file with OPENAI_API_KEY=your-key-here")
        return
    
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Or create a .env file with ANTHROPIC_API_KEY=your-key-here")
        return
    
    # Initialize chatbot
    try:
        chatbot = Chatbot(provider=provider, model=model)
        print(f"\n✅ Chatbot initialized with {provider.upper()} ({model})")
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        return
    
    # Optional system message
    system_msg = input("\nEnter system message (optional, press Enter to skip): ").strip()
    if system_msg:
        chatbot.add_message("system", system_msg)
        print(f"System message set: {system_msg[:50]}...")
    
    print("\n" + "=" * 60)
    print("Chat started! Type 'exit', 'quit', or 'clear' to clear history.")
    print("=" * 60)
    print()
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "clear":
                chatbot.clear_history()
                print("\n✅ Conversation history cleared.")
                continue
            
            if user_input.lower() == "history":
                history = chatbot.get_history()
                print("\n📜 Conversation History:")
                for i, msg in enumerate(history):
                    role = msg["role"].upper()
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    print(f"{i+1}. {role}: {content}")
                continue
            
            # Get and display streaming response
            print("\nAssistant: ", end="", flush=True)
            stream = chatbot.get_streaming_response(user_input)
            print_streaming_response(stream)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...")


if __name__ == "__main__":
    main()