
#setx OPENAI_API_KEY "your_api_key_here"


from openai import OpenAI
client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")