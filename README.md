# AI Application Study

A study project for learning and experimenting with AI application development using various AI APIs and frameworks.

## Project Structure

```
AI-Application-Study/
├── Week1/                    # Week 1 study materials
│   ├── DeepSeekAPIDemo.py    # DeepSeek API demonstration
│   ├── OpenAIDemo.py         # OpenAI API demonstration
│   └── tokens-context-study/ # Interactive React app for learning tokens & context windows
│       ├── src/              # React source code
│       ├── public/           # Static assets
│       ├── package.json      # Node.js dependencies
│       └── README.md         # React app documentation
├── .gitignore               # Git ignore files
└── README.md                # This file
```

## Getting Started

### Prerequisites
- Python 3.7+ (for API demos)
- Node.js 18+ (for React app)
- Git
- Required Python packages (install via pip):
  ```bash
  pip install openai
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone git@github.com:fromnewcoder/AI-Application-Study.git
   cd AI-Application-Study
   ```

2. Set up your API keys:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     DEEPSEEK_API_KEY=your_deepseek_api_key_here
     ```

## Usage

### Week 1 Demos

#### Python API Demos
- **OpenAIDemo.py**: Basic OpenAI API usage example
- **DeepSeekAPIDemo.py**: DeepSeek API integration example

Run the demos:
```bash
cd Week1
python OpenAIDemo.py
python DeepSeekAPIDemo.py
```

#### Interactive Tokens & Context Windows Study App
A comprehensive React application for learning about tokenization, context windows, and LLM architecture:

```bash
cd Week1/tokens-context-study
npm install
npm run dev
```

The app includes:
- **Tokenization Visualizer**: See how text gets split into tokens
- **tiktoken Tutorial**: Learn to install and use OpenAI's tokenizer
- **Token Counting Functions**: Production-ready Python code for counting tokens
- **Context Window Architecture**: Decision tree for choosing the right approach
- **Interactive Quizzes**: Test your understanding of key concepts

## Project Goals
- Learn AI API integration
- Experiment with different AI models
- Build practical AI applications
- Document learning progress
- Create interactive educational tools

## Contributing
This is a personal study project. Feel free to fork and adapt for your own learning.

## License
This project is for educational purposes.
