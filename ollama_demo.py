from ollama import chat

response = chat(
    model='gemma4:e4b-it-q8_0',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)