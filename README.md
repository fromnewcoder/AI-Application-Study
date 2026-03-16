# AI Application Study

A study project for learning and experimenting with AI application development using various AI APIs and frameworks.

## Project Structure

```
AI-Application-Study/
├── Week1/                           # Week 1 study materials
│   ├── DeepSeekAPIDemo.py          # DeepSeek API demonstration
│   ├── OpenAIDemo.py               # OpenAI API demonstration
│   ├── MiniMaxDemo.py              # MiniMax API demonstration
│   ├── PromptTemplate.py           # Prompt engineering templates
│   ├── TokenAndConextWindows.py    # Token & context window education
│   ├── tokens-context-study/        # Interactive React app for learning tokens & context windows
│   │   ├── src/                    # React source code
│   │   ├── public/                 # Static assets
│   │   ├── package.json            # Node.js dependencies
│   │   └── README.md               # React app documentation
│   ├── CLI Chatbot/                # CLI-based chatbot implementations
│   │   ├── From ClaudeAI/          # Claude AI chatbot
│   │   └── From Google AI/          # Google AI chatbot
│   ├── prompt_engineering_basics.docx    # Prompt engineering basics guide
│   ├── prompt_engineering_advanced.docx  # Advanced prompt engineering
│   └── prompt_engineering_III.docx       # Prompt engineering Level III
├── Week2/                           # Week 2 study materials
│   ├── EmbeddingsTheory.md         # Embeddings theory guide
│   ├── EmbeddingsDemo.py           # Embeddings demo with similarity matrix
│   ├── ChromaDBTheory.md           # Vector databases & Chroma DB guide
│   ├── ChromaDBDemo.py             # Store & query 100 embeddings
│   ├── RAGTheory.md               # RAG architecture & chunking guide
│   ├── RAGDemo.py                 # RAG pipeline demo
│   └── requirements.txt            # Python dependencies
├── Week3/                           # Week 3 study materials
│   ├── StructuredOutputsDemo.py   # JSON mode, Pydantic, instructor
│   └── README.md                   # Structured outputs guide
├── .gitignore                      # Git ignore files
└── README.md                        # This file
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
     MINIMAX_API_KEY=your_minimax_api_key_here
     ```

## Usage

### Week 1 Demos

#### Python API Demos
- **OpenAIDemo.py**: Basic OpenAI API usage example
- **DeepSeekAPIDemo.py**: DeepSeek API integration example
- **MiniMaxDemo.py**: MiniMax API integration example
- **PromptTemplate.py**: Prompt engineering template examples
- **TokenAndConextWindows.py**: Token & context window educational content

Run the demos:
```bash
cd Week1
python OpenAIDemo.py
python DeepSeekAPIDemo.py
python MiniMaxDemo.py
```

#### CLI Chatbots
Terminal-based chatbot implementations using different AI providers:

- **From ClaudeAI/**: CLI chatbot using Anthropic's Claude
- **From Google AI/**: CLI chatbot using Google AI

Run a chatbot:
```bash
cd Week1/CLI Chatbot/From ClaudeAI
pip install -r requirements.txt
python chatbot.py
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

### Week 2 Demos

#### Embeddings Theory & Demo
Learn about text embeddings, vector spaces, and cosine similarity:

- **EmbeddingsTheory.md**: Theory guide covering what embeddings are, vector space concepts, cosine similarity, how embedding models differ, and use cases
- **EmbeddingsDemo.py**: Interactive demo that embeds 10 sentences and computes a similarity matrix

Run the demo:
```bash
cd Week2
pip install -r requirements.txt
python -m spacy download en_core_web_md
python EmbeddingsDemo.py
```

The demo uses spaCy's word vectors (no API key required) to demonstrate semantic similarity between sentences.

#### Vector Databases (Chroma DB)
Learn about vector databases, CRUD operations, indexing, similarity search, and metadata filtering:

- **ChromaDBTheory.md**: Theory guide covering what vector databases are, Chroma DB setup, CRUD operations, indexing (HNSW), similarity search, and metadata filtering
- **ChromaDBDemo.py**: Interactive demo that stores and queries 100 embeddings across 10 categories

Run the demo:
```bash
cd Week2
pip install -r requirements.txt
python -m spacy download en_core_web_md
python ChromaDBDemo.py
```

The demo includes:
- Adding 100 documents across 10 categories (technology, food, sports, etc.)
- Basic similarity search queries
- Metadata filtering by category and word count
- Combined semantic search + metadata filter
- CRUD operations

#### RAG Architecture Deep Dive
Learn about Retrieval-Augmented Generation architecture, chunking strategies, and augmented prompting:

- **RAGTheory.md**: Theory guide covering RAG fundamentals, chunking strategies (fixed, semantic, sliding), overlap, retrieval pipeline design, and augmented prompting
- **RAGDemo.py**: Interactive demo that demonstrates all chunking strategies and full RAG pipeline with Chroma

Run the demo:
```bash
cd Week2
pip install -r requirements.txt
python -m spacy download en_core_web_md
python RAGDemo.py
```

The demo includes:
- Three chunking strategies: fixed-size, semantic (by sentence), sliding window
- Vector storage and semantic retrieval
- Augmented prompting examples
- Decision diagram for choosing chunking strategies
- Metadata filtering by category and word count
- Combined semantic search + metadata filter
- Get document by ID
- Update and delete operations

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
