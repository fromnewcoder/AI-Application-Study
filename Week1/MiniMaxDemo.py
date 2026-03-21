
# 推荐使用 Anthropic API 兼容
#export ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
#export ANTHROPIC_API_KEY=${YOUR_API_KEY}
#$env:ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
#$env:ANTHROPIC_API_KEY=${YOUR_API_KEY}

import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="MiniMax-M2.7",
    max_tokens=1000,
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hi, how are you?"
                }
            ]
        }
    ]
)

for block in message.content:
    if block.type == "thinking":
        print(f"Thinking:\n{block.thinking}\n")
    elif block.type == "text":
        print(f"Text:\n{block.text}\n")