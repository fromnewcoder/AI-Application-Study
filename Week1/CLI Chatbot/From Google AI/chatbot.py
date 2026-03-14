import os
import sys
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

def main():
    # Initialize the client (automatically looks for OPENAI_API_KEY env var)
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )

    # Conversation Management: Store message history in a local list
    # We include a system message to define the assistant's behavior
    messages = [
        {"role": "system", "content": "You are a helpful and concise software engineering assistant."}
    ]

    print("--- Streaming CLI Chatbot (Type 'exit' or 'quit' to stop) ---")

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # CLI Exit Logic
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            # Append user message to history to maintain multi-turn context
            messages.append({"role": "user", "content": user_input})

            # API Interaction: Direct call with stream=True
            stream = client.chat.completions.create(
                model="deepseek-chat",  # or "gpt-3.5-turbo"
                messages=messages,
                stream=True
            )

            print("Assistant: ", end="", flush=True)
            full_response_content = ""

            # Streaming Logic: Generator-based iteration over response chunks
            for chunk in stream:
                # Extract text delta from the chunk
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    full_response_content += delta
            
            print() # New line after stream finishes

            # Update history with the assistant's final response for next turn context
            messages.append({"role": "assistant", "content": full_response_content})

        except RateLimitError:
            # Handle 429 errors (too many requests)
            print("\n[Error] Rate limit exceeded. Please wait a moment before trying again.")
        except APIConnectionError:
            # Handle network-related issues
            print("\n[Error] Connection failed. Check your internet or API proxy settings.")
        except APIError as e:
            # Handle generic API issues (invalid keys, server errors)
            print(f"\n[Error] An API error occurred: {e}")
        except KeyboardInterrupt:
            # Graceful exit on Ctrl+C
            print("\nInterrupted by user. Exiting...")
            break

if __name__ == "__main__":
    main()
