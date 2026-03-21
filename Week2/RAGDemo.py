"""
RAG Pipeline Demo: Demonstrates chunking strategies and retrieval
"""

import spacy
import chromadb
from typing import List, Tuple

# Load spaCy for semantic operations
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")

# Sample document (simulating a longer document)
SAMPLE_DOCUMENT = """
Artificial Intelligence: A Comprehensive Overview

Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

Machine Learning Fundamentals

Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.

Deep Learning Architecture

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs.

Natural Language Processing

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them.

Computer Vision Applications

Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information.
"""

# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def fixed_chunking(text: str, chunk_size: int = 200) -> List[str]:
    """Fixed-size character chunking (non-overlapping)."""
    if not text or chunk_size <= 0:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def sliding_window_chunking(text: str, window_size: int = 200, step: int = 100) -> List[str]:
    """Sliding window chunking with overlap (step < window_size creates overlap)."""
    if not text or window_size <= 0:
        return [text] if text else []

    chunks = []

    for start in range(0, len(text), step):
        end = min(start + window_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Stop if we've reached the end
        if end >= len(text):
            break

    return chunks


def semantic_chunking(text: str, max_sentences: int = 2) -> List[str]:
    """Split by sentences using spaCy."""
    doc = nlp(text)
    chunks = []
    current_chunk = []

    for sent in doc.sents:
        current_chunk.append(sent.text)

        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def recursive_chunking(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Simple recursive chunking trying natural boundaries."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at paragraph boundary first
        if "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size // 3:
                chunk = chunk[:last_break]
        # Then sentence boundary
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += len(chunk) - overlap
        if start >= len(text):
            break

    return chunks


# =============================================================================
# DEMONSTRATE CHUNKING STRATEGIES
# =============================================================================

print("=" * 80)
print("RAG PIPELINE: CHUNKING STRATEGIES DEMO")
print("=" * 80)

print(f"\nSource document length: {len(SAMPLE_DOCUMENT)} characters\n")

# Strategy 1: Fixed Chunking
print("-" * 40)
print("1. Fixed-Size Chunking (200 chars, non-overlapping)")
print("-" * 40)
fixed_chunks = fixed_chunking(SAMPLE_DOCUMENT, chunk_size=200)
for i, chunk in enumerate(fixed_chunks):
    print(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:60]}...")
print(f"  Total chunks: {len(fixed_chunks)}")

# Strategy 2: Semantic Chunking
print("\n" + "-" * 40)
print("2. Semantic Chunking (by sentences, max 2)")
print("-" * 40)
semantic_chunks = semantic_chunking(SAMPLE_DOCUMENT, max_sentences=2)
for i, chunk in enumerate(semantic_chunks):
    print(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:60]}...")
print(f"  Total chunks: {len(semantic_chunks)}")

# Strategy 3: Sliding Window (replacing recursive for simplicity)
print("\n" + "-" * 40)
print("3. Sliding Window Chunking (200 window, 100 step)")
print("-" * 40)
sliding_chunks = sliding_window_chunking(SAMPLE_DOCUMENT, window_size=200, step=100)
for i, chunk in enumerate(sliding_chunks):
    print(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:60]}...")
print(f"  Total chunks: {len(sliding_chunks)}")

# =============================================================================
# RAG PIPELINE WITH CHROMA
# =============================================================================

print("\n" + "=" * 80)
print("RAG PIPELINE: RETRIEVAL & AUGMENTATION")
print("=" * 80)

# Initialize Chroma
print("\nInitializing Chroma DB...")
client = chromadb.PersistentClient(path="./rag_demo_db")

# Clear previous
try:
    client.delete_collection("rag_docs")
except:
    pass

collection = client.create_collection("rag_docs")

# Use semantic chunks for embedding
chunks_to_embed = semantic_chunks

# Generate embeddings using Chroma's built-in model
print(f"\nEmbedding {len(chunks_to_embed)} chunks...")

collection.add(
    documents=chunks_to_embed,
    ids=[f"chunk_{i}" for i in range(len(chunks_to_embed))],
    metadatas=[{"index": i} for i in range(len(chunks_to_embed))]
)

print(f"Stored {collection.count()} documents in vector DB")

# =============================================================================
# RETRIEVAL DEMOS
# =============================================================================

def display_results(query: str, results: dict):
    """Display retrieval results."""
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance

        print(f"  {i+1}. [{doc_id}] (similarity: {similarity:.3f})")
        print(f"     {doc[:100]}...")

    print()


# Demo 1: Query about Machine Learning
print("\n" + "=" * 80)
print("DEMO 1: Query about Machine Learning")
print("=" * 80)
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)
display_results("What is machine learning?", results)

# Demo 2: Query about Deep Learning
print("\n" + "=" * 80)
print("DEMO 2: Query about Deep Learning")
print("=" * 80)
results = collection.query(
    query_texts=["neural networks deep learning"],
    n_results=2
)
display_results("neural networks deep learning", results)

# Demo 3: Query about NLP
print("\n" + "=" * 80)
print("DEMO 3: Query about Natural Language Processing")
print("=" * 80)
results = collection.query(
    query_texts=["language processing computers"],
    n_results=2
)
display_results("language processing computers", results)

# =============================================================================
# AUGMENTED PROMPTING DEMO
# =============================================================================

print("\n" + "=" * 80)
print("DEMO 4: AUGMENTED PROMPTING")
print("=" * 80)

def build_augmented_prompt(query: str, retrieved_docs: List[str], max_context: int = 500) -> str:
    """Build augmented prompt with retrieved context."""
    context_parts = []
    current_len = 0

    for doc in retrieved_docs:
        if current_len + len(doc) > max_context:
            break
        context_parts.append(doc)
        current_len += len(doc)

    context = "\n\n".join(context_parts)

    prompt = f"""Use the following context to answer the question. If the answer
is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    return prompt


# Get context for a query
results = collection.query(
    query_texts=["What is artificial intelligence?"],
    n_results=2
)

retrieved_docs = results['documents'][0]

# Build augmented prompt
prompt = build_augmented_prompt(
    "What is artificial intelligence?",
    retrieved_docs,
    max_context=500
)

print("\nAugmented Prompt:")
print("=" * 50)
print(prompt)
print("=" * 50)
print("\n(Note: This is the prompt that would be sent to an LLM)")
print("The LLM would generate a response based on the retrieved context)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
RAG Pipeline Components Demonstrated:

1. CHUNKING STRATEGIES:
   - Fixed-size: Simple, predictable, may break semantic units
   - Semantic: Preserves sentence boundaries, more natural chunks
   - Recursive: Tries multiple separators, better structure

2. EMBEDDING & STORAGE:
   - Each chunk is embedded and stored in vector DB (Chroma)
   - Metadata tracks chunk index for reference

3. RETRIEVAL:
   - Semantic similarity search finds relevant chunks
   - Can filter by metadata if needed

4. AUGMENTATION:
   - Retrieved context is formatted into prompt
   - Includes instructions for LLM to use context
   - Can limit context length to fit token limits

Key Trade-offs:
- Smaller chunks = more precise retrieval, but less context
- Larger chunks = more context, but may include noise
- Overlap helps maintain context across boundaries
""")

print("\nDemo complete! Vector DB saved to: ./rag_demo_db")
