"""
RAG PDF Demo: Demonstrates PDF ingestion pipeline
Load PDF -> Chunk text -> Generate embeddings -> Store in Chroma with metadata
Integrated with MiniMax and DeepSeek LLMs for answer generation
"""

import os
import spacy
import chromadb
from typing import List, Tuple
import PyPDF2
import anthropic
from openai import OpenAI

# Load spaCy for semantic operations
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")

# Initialize LLM clients
minimax_client = None
deepseek_client = None

print("Initializing MiniMax client...")
try:
    minimax_client = anthropic.Anthropic()
    print("  MiniMax client ready")
except Exception as e:
    print(f"  Warning: Could not initialize MiniMax client: {e}")

print("Initializing DeepSeek client...")
try:
    deepseek_client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )
    print("  DeepSeek client ready")
except Exception as e:
    print(f"  Warning: Could not initialize DeepSeek client: {e}")

# Default LLM
DEFAULT_LLM = "minimax"


# =============================================================================
# LLM INTEGRATION
# =============================================================================

def generate_answer(query: str, context_docs: List[str], max_context: int = 1500,
                    llm: str = DEFAULT_LLM) -> str:
    """
    Generate answer using MiniMax or DeepSeek LLM with retrieved context.

    Args:
        query: User question
        context_docs: List of retrieved document chunks
        max_context: Maximum characters to include in context
        llm: LLM to use - "minimax" (default) or "deepseek"
    """
    # Build context from retrieved docs
    context_parts = []
    current_len = 0

    for doc in context_docs:
        if current_len + len(doc) > max_context:
            break
        context_parts.append(doc)
        current_len += len(doc)

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""Use the context below to answer the question. If the answer is not in the context, say "I don't have enough information from the document."

Context:
{context}

Question: {query}

Answer:"""

    if llm == "deepseek":
        return _generate_deepseek(prompt)
    else:
        return _generate_minimax(prompt)


def _generate_minimax(prompt: str) -> str:
    """Generate answer using MiniMax."""
    if not minimax_client:
        return "Error: MiniMax client not initialized"

    try:
        message = minimax_client.messages.create(
            model="MiniMax-M2.5",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        )

        for block in message.content:
            if block.type == "text":
                return block.text
            elif block.type == "thinking":
                return block.thinking

        return "No response generated"

    except Exception as e:
        return f"Error generating answer (MiniMax): {e}"


def _generate_deepseek(prompt: str) -> str:
    """Generate answer using DeepSeek."""
    if not deepseek_client:
        return "Error: DeepSeek client not initialized"

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating answer (DeepSeek): {e}"


# =============================================================================
# PDF LOADING
# =============================================================================

def load_pdf(pdf_path: str) -> str:
    """Load text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    return text


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

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# =============================================================================
# RAG PIPELINE
# =============================================================================

def run_rag_pipeline(pdf_path: str, chunking_strategy: str = "semantic",
                     chunk_size: int = 200, step: int = 100, max_sentences: int = 2):
    """
    Run complete RAG pipeline: PDF -> Chunk -> Embed -> Store

    Args:
        pdf_path: Path to PDF file
        chunking_strategy: "fixed", "sliding", or "semantic"
        chunk_size: For fixed/sliding chunking
        step: For sliding window
        max_sentences: For semantic chunking
    """
    print("=" * 80)
    print("RAG PDF INGESTION PIPELINE")
    print("=" * 80)

    # Step 1: Load PDF
    print(f"\n[1] Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        return None

    text = load_pdf(pdf_path)
    print(f"    Extracted {len(text)} characters from PDF")

    # Step 2: Chunk text
    print(f"\n[2] Chunking text using '{chunking_strategy}' strategy")
    if chunking_strategy == "fixed":
        chunks = fixed_chunking(text, chunk_size=chunk_size)
    elif chunking_strategy == "sliding":
        chunks = sliding_window_chunking(text, window_size=chunk_size, step=step)
    else:  # semantic
        chunks = semantic_chunking(text, max_sentences=max_sentences)

    print(f"    Created {len(chunks)} chunks")

    # Show sample chunks
    print("\n    Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"      Chunk {i}: [{len(chunk)} chars] {chunk[:80]}...")

    # Step 3: Initialize Chroma and store
    print("\n[3] Generating embeddings and storing in Chroma...")
    client = chromadb.PersistentClient(path="./rag_pdf_db")

    # Clear previous
    try:
        client.delete_collection("pdf_chunks")
    except:
        pass

    collection = client.create_collection("pdf_chunks")

    # Add chunks with metadata
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "chunk_id": i,
            "source": pdf_path,
            "strategy": chunking_strategy,
            "char_count": len(chunk)
        })

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=metadatas
    )

    print(f"    Stored {collection.count()} chunks in vector DB")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Vector DB saved to: ./rag_pdf_db")
    print(f"Total chunks indexed: {len(chunks)}")

    return collection


# =============================================================================
# RETRIEVAL DEMO
# =============================================================================

def demo_retrieval(collection, query: str, n_results: int = 3, use_llm: bool = True,
                   llm: str = DEFAULT_LLM):
    """Demo retrieval and LLM-generated answer from the vector DB."""
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Show retrieved chunks
    print("\nRetrieved Context:")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance

        print(f"  {i+1}. [{doc_id}] (similarity: {similarity:.3f})")
        print(f"     {doc[:150]}...")

    # Generate answer with LLM
    if use_llm:
        llm_name = "MiniMax" if llm == "minimax" else "DeepSeek"
        print("\n" + "-" * 50)
        print(f"LLM Answer ({llm_name}):")
        print("-" * 50)

        context_docs = results['documents'][0]
        answer = generate_answer(query, context_docs, llm=llm)
        print(f"  {answer}")

    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Parse arguments
    # Usage: python RAGPDFDemo.py [pdf_path] [--llm minimax|deepseek]
    PDF_PATH = "AI_Study_Material.pdf"
    llm = DEFAULT_LLM

    if len(sys.argv) > 1:
        if sys.argv[1] == "--llm" and len(sys.argv) > 2:
            llm = sys.argv[2]
            if len(sys.argv) > 3:
                PDF_PATH = sys.argv[3]
        else:
            PDF_PATH = sys.argv[1]

    # Run pipeline with semantic chunking (best for preserving meaning)
    collection = run_rag_pipeline(
        pdf_path=PDF_PATH,
        chunking_strategy="semantic",
        max_sentences=2
    )

    if collection:
        # Demo retrieval with default LLM (MiniMax)
        print("\n" + "=" * 80)
        print("RETRIEVAL DEMO (LLM: " + ("MiniMax" if llm == "minimax" else "DeepSeek") + ")")
        print("=" * 80)

        # To use DeepSeek: pass llm="deepseek" to demo_retrieval
        # demo_retrieval(collection, query, llm="deepseek")
        demo_retrieval(collection, "What is machine learning?", llm=llm)
        demo_retrieval(collection, "Explain deep learning neural networks", llm=llm)
        demo_retrieval(collection, "Natural language processing applications", llm=llm)

        # Example: Uncomment to test DeepSeek
        # print("\n" + "=" * 80)
        # print("RETRIEVAL DEMO (DeepSeek)")
        # print("=" * 80)
        # demo_retrieval(collection, "What is machine learning?", llm="deepseek")
