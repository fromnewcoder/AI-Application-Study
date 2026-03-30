const Anthropic = require('@anthropic-ai/sdk');

const anthropic = new Anthropic();

// ============================================================
// 1. Tool schema definition
// ============================================================
const tools = [
  {
    name: "get_weather",
    description: "Get current weather for a city. Call this whenever the user asks about weather, temperature, or forecast.",
    input_schema: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "City name, e.g. 'Singapore' or 'Paris, France'"
        },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"],
          description: "Temperature unit. Default celsius."
        }
      },
      required: ["location"]
    }
  }
];

// Simulated weather API
async function callWeatherAPI(location, unit = "celsius") {
  // In production, call a real weather service
  const mockData = {
    "Singapore": { temp: 31, condition: "Humid and hazy", humidity: 84 },
    "Tokyo": { temp: 18, condition: "Partly cloudy", humidity: 62 },
    "London": { temp: 9, condition: "Overcast", humidity: 78 },
    "Sydney": { temp: 24, condition: "Clear skies", humidity: 55 }
  };
  const weather = mockData[location] || { temp: 20, condition: "Unknown", humidity: 50 };
  if (unit === "fahrenheit") {
    weather.temp = Math.round(weather.temp * 9/5 + 32);
  }
  return weather;
}

// ============================================================
// Main function: handle a weather query with tool calling
// ============================================================
async function askWeather(question) {
  // Step 1: Send request with tools attached
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1000,
    tools: tools,
    messages: [
      { role: "user", content: question }
    ]
  });

  console.log("Response stop_reason:", response.stop_reason);
  console.log("Response content:", JSON.stringify(response.content, null, 2));

  // Step 2: Check if model wants to call a tool
  const toolUseBlock = response.content.find(
    block => block.type === "tool_use"
  );

  if (!toolUseBlock) {
    // No tool call needed, return the text response
    const textBlock = response.content.find(block => block.type === "text");
    return textBlock?.text || "No response";
  }

  // Step 3: Parse the tool_use block
  const { id: toolUseId, name, input } = toolUseBlock;
  console.log(`\nTool called: ${name}`);
  console.log(`Input:`, input);

  // Step 4: Execute the tool
  let toolResult;
  if (name === "get_weather") {
    toolResult = await callWeatherAPI(input.location, input.unit);
    console.log(`Weather result:`, toolResult);
  } else {
    toolResult = { error: `Unknown tool: ${name}` };
  }

  // Step 5: Send tool_result back to the model
  const finalResponse = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1000,
    tools: tools,
    messages: [
      { role: "user", content: question },
      { role: "assistant", content: response.content },
      {
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: toolUseId,
          content: JSON.stringify(toolResult)
        }]
      }
    ]
  });

  // Step 6: Return the final answer
  const textBlock = finalResponse.content.find(block => block.type === "text");
  return textBlock?.text || "No response";
}

// Run the demo
askWeather("What's the weather in Singapore?")
  .then(answer => {
    console.log("\n=== Final Answer ===");
    console.log(answer);
  })
  .catch(console.error);