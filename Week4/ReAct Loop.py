import anthropic
import os

client = anthropic.Anthropic()
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
            },
            "required": ["location"]
        }
    }
]

# Simple mock tool execution
def execute_tool(tool_name, tool_input):
    if tool_name == "get_weather":
        return f"The weather in {tool_input['location']} is 22°C and sunny."
    return "Tool not found"

def run_react_agent(user_query, max_iterations=5):
    messages = [{"role": "user", "content": user_query}]
    
    for i in range(max_iterations):
        # 1. Reason: Send messages and tools to Claude
        response = client.messages.create(
            model="MiniMax-M2.7",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Add Claude's reasoning/response to message history
        messages.append({"role": "assistant", "content": response.content})
        
        # 2. Check for Action: Did Claude call a tool?
        tool_use = next((content for content in response.content if content.type == "tool_use"), None)
        
        if not tool_use:
            # Final Answer reached - find the text block
            text_block = next((content for content in response.content if content.type == "text"), None)
            return text_block.text if text_block else "No text block found"

        # 3. Observe: Execute the tool and return the result
        tool_result = execute_tool(tool_use.name, tool_use.input)
        messages.append({
            "role": "user", 
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": tool_result
                }
            ]
        })
        print(messages)
        print(f"Iteration {i+1}: Called {tool_use.name} -> {tool_result}")

    return "Max iterations reached without a final answer."


if __name__ == "__main__":
    run_react_agent("what's the weather in San Francisco")