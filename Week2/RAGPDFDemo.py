"""
RAG PDF Demo: Demonstrates PDF ingestion pipeline
Load PDF -> Chunk text -> Generate embeddings -> Store in Chroma with metadata
Integrated with MiniMax LLM for answer generation
"""

import os
import spacy
import chromadb
from typing import List, Tuple
import PyPDF2
import anthropic

# Load spaCy for semantic operations
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")

# Initialize MiniMax client
print("Initializing MiniMax client...")
try:
    client = anthropic.Anthropic()
    print("MiniMax client ready")
except Exception as e:
    print(f"Warning: Could not initialize MiniMax client: {e}")
    client = None


# =============================================================================
# LLM INTEGRATION
# =============================================================================

def generate_answer(query: str, context_docs: List[str], max_context: int = 1500) -> str:
    """
    Generate answer using MiniMax LLM with retrieved context.

    Args:
        query: User question
        context_docs: List of retrieved document chunks
        max_context: Maximum characters to include in context
    """
    if not client:
        return "Error: MiniMax client not initialized"

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

    try:
        message = client.messages.create(
            model="MiniMax-M2.5",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        )

        # Extract response
        for block in message.content:
            if block.type == "text":
                return block.text
            elif block.type == "thinking":
                return block.thinking

        return "No response generated"

    except Exception as e:
        return f"Error generating answer: {e}"


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

def demo_retrieval(collection, query: str, n_results: int = 3, use_llm: bool = True):
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
        print("\n" + "-" * 50)
        print("LLM Answer (MiniMax):")
        print("-" * 50)

        context_docs = results['documents'][0]
        answer = generate_answer(query, context_docs)
        print(f"  {answer}")

    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Default PDF path - can be overridden via command line
    PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "AI_Study_Material.pdf"

    # Run pipeline with semantic chunking (best for preserving meaning)
    collection = run_rag_pipeline(
        pdf_path=PDF_PATH,
        chunking_strategy="semantic",
        max_sentences=2
    )

    if collection:
        # Demo retrieval
        print("\n" + "=" * 80)
        print("RETRIEVAL DEMO")
        print("=" * 80)

        demo_retrieval(collection, "What is machine learning?")
        demo_retrieval(collection, "Explain deep learning neural networks")
        demo_retrieval(collection, "Natural language processing applications")
