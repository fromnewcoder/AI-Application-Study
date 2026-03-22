"""
RAG PDF Demo: Demonstrates PDF ingestion pipeline
Load PDF -> Chunk text -> Generate embeddings -> Store in Chroma with metadata
Integrated with MiniMax and DeepSeek LLMs for answer generation
Production-ready with: structured output, streaming, retry logic, logging
"""

import os
import sys
import time
import json
import logging
from functools import wraps
from typing import List, Tuple, Optional, Dict, Any, Iterator

import spacy
import chromadb
import PyPDF2
import anthropic
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAGPDFDemo")

# Load spaCy for semantic operations
logger.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")

# Initialize LLM clients
minimax_client = None
deepseek_client = None

logger.info("Initializing MiniMax client...")
try:
    minimax_client = anthropic.Anthropic()
    logger.info("  MiniMax client ready")
except Exception as e:
    logger.warning(f"  Could not initialize MiniMax client: {e}")

logger.info("Initializing DeepSeek client...")
try:
    deepseek_client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )
    logger.info("  DeepSeek client ready")
except Exception as e:
    logger.warning(f"  Could not initialize DeepSeek client: {e}")

# Default LLM
DEFAULT_LLM = "minimax"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def with_retry(func):
    """Decorator that retries a function with exponential backoff on failure."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                else:
                    logger.error(f"{func.__name__} failed after {MAX_RETRIES} attempts: {e}")

        raise last_exception
    return wrapper


# =============================================================================
# STRUCTURED OUTPUT
# =============================================================================

class StructuredOutput:
    """Structured output for RAG responses with metadata."""

    def __init__(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        query: str,
        llm: str,
        streaming: bool = False,
        chunks_used: int = 0,
        total_chunks_available: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.answer = answer
        self.sources = sources
        self.query = query
        self.llm = llm
        self.streaming = streaming
        self.chunks_used = chunks_used
        self.total_chunks_available = total_chunks_available
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "llm": self.llm,
            "streaming": self.streaming,
            "chunks_used": self.chunks_used,
            "total_chunks_available": self.total_chunks_available,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return self.to_json()


# =============================================================================
# LLM INTEGRATION
# =============================================================================


# =============================================================================
# LLM INTEGRATION
# =============================================================================

def generate_answer(query: str, context_docs: List[str], max_context: int = 1500,
                    llm: str = DEFAULT_LLM, structured: bool = True,
                    return_sources: bool = True) -> StructuredOutput:
    """
    Generate answer using MiniMax or DeepSeek LLM with retrieved context.

    Args:
        query: User question
        context_docs: List of retrieved document chunks
        max_context: Maximum characters to include in context
        llm: LLM to use - "minimax" (default) or "deepseek"
        structured: If True, return StructuredOutput; if False, return raw string
        return_sources: Include source chunks in structured output
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
        answer = _generate_deepseek_with_retry(prompt)
    else:
        answer = _generate_minimax_with_retry(prompt)

    if structured:
        sources = []
        if return_sources:
            for i, doc in enumerate(context_docs[:5]):  # Limit to 5 sources
                sources.append({
                    "chunk_id": i,
                    "text_preview": doc[:200],
                    "char_count": len(doc)
                })

        return StructuredOutput(
            answer=answer,
            sources=sources,
            query=query,
            llm=llm,
            streaming=False,
            chunks_used=len(context_parts),
            metadata={"model": llm}
        )

    return answer


@with_retry
def _generate_minimax_with_retry(prompt: str) -> str:
    """Generate answer using MiniMax with retry logic."""
    if not minimax_client:
        return "Error: MiniMax client not initialized"

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


@with_retry
def _generate_deepseek_with_retry(prompt: str) -> str:
    """Generate answer using DeepSeek with retry logic."""
    if not deepseek_client:
        return "Error: DeepSeek client not initialized"

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


def stream_generate_answer(query: str, context_docs: List[str], max_context: int = 1500,
                           llm: str = DEFAULT_LLM) -> Iterator[Tuple[str, StructuredOutput]]:
    """
    Stream answer using MiniMax or DeepSeek LLM with retrieved context.

    Args:
        query: User question
        context_docs: List of retrieved document chunks
        max_context: Maximum characters to include in context
        llm: LLM to use - "minimax" (default) or "deepseek"

    Yields:
        Tuple of (streamed_chunk, StructuredOutput) where the structured output
        is yielded once at the end with the full answer
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

    full_answer = ""

    if llm == "deepseek":
        # DeepSeek streaming
        if not deepseek_client:
            yield ("Error: DeepSeek client not initialized", None)
            return

        try:
            stream = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_answer += text_chunk
                    yield (text_chunk, None)

        except Exception as e:
            yield (f"Error streaming (DeepSeek): {e}", None)
            return
    else:
        # MiniMax streaming
        if not minimax_client:
            yield ("Error: MiniMax client not initialized", None)
            return

        try:
            with minimax_client.messages.stream(
                model="MiniMax-M2.5",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            ) as stream:
                for chunk in stream:
                    if chunk.type == "text" and chunk.text:
                        full_answer += chunk.text
                        yield (chunk.text, None)
                    elif chunk.type == "thinking_block" and chunk.thinking:
                        full_answer += chunk.thinking
                        yield (chunk.thinking, None)

        except Exception as e:
            yield (f"Error streaming (MiniMax): {e}", None)
            return

    # Build sources
    sources = []
    for i, doc in enumerate(context_docs[:5]):
        sources.append({
            "chunk_id": i,
            "text_preview": doc[:200],
            "char_count": len(doc)
        })

    # Yield final structured output
    structured_output = StructuredOutput(
        answer=full_answer,
        sources=sources,
        query=query,
        llm=llm,
        streaming=True,
        chunks_used=len(context_parts),
        metadata={"model": llm, "stream_completed": True}
    )

    yield ("", structured_output)


# =============================================================================
# PDF LOADING
# =============================================================================

@with_retry
def load_pdf(pdf_path: str) -> str:
    """Load text from a PDF file with retry logic."""
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
    logger.info("=" * 80)
    logger.info("RAG PDF INGESTION PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load PDF
    logger.info(f"[1] Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return None

    text = load_pdf(pdf_path)
    logger.info(f"Extracted {len(text)} characters from PDF")

    # Step 2: Chunk text
    logger.info(f"[2] Chunking text using '{chunking_strategy}' strategy")
    if chunking_strategy == "fixed":
        chunks = fixed_chunking(text, chunk_size=chunk_size)
    elif chunking_strategy == "sliding":
        chunks = sliding_window_chunking(text, window_size=chunk_size, step=step)
    else:  # semantic
        chunks = semantic_chunking(text, max_sentences=max_sentences)

    logger.info(f"Created {len(chunks)} chunks")

    # Show sample chunks
    logger.debug("Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        logger.debug(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:80]}...")

    # Step 3: Initialize Chroma and store
    logger.info("[3] Generating embeddings and storing in Chroma...")
    client = chromadb.PersistentClient(path="./rag_pdf_db")

    # Clear previous
    try:
        client.delete_collection("pdf_chunks")
    except Exception:
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

    logger.info(f"Stored {collection.count()} chunks in vector DB")

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Vector DB saved to: ./rag_pdf_db")
    logger.info(f"Total chunks indexed: {len(chunks)}")

    return collection


# =============================================================================
# RETRIEVAL DEMO
# =============================================================================

def demo_retrieval(collection, query: str, n_results: int = 3, use_llm: bool = True,
                   llm: str = DEFAULT_LLM, use_streaming: bool = False):
    """
    Demo retrieval and LLM-generated answer from the vector DB.

    Args:
        collection: Chroma collection
        query: User question
        n_results: Number of chunks to retrieve
        use_llm: Whether to generate LLM answer
        llm: LLM to use - "minimax" or "deepseek"
        use_streaming: If True, use streaming mode
    """
    logger.info(f"Query: '{query}'")
    logger.debug("-" * 50)

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Show retrieved chunks
    logger.info("Retrieved Context:")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance

        logger.info(f"  {i+1}. [{doc_id}] (similarity: {similarity:.3f})")
        logger.debug(f"     {doc[:150]}...")

    # Generate answer with LLM
    if use_llm:
        llm_name = "MiniMax" if llm == "minimax" else "DeepSeek"
        logger.info("-" * 50)
        logger.info(f"LLM Answer ({llm_name}):")
        logger.info("-" * 50)

        context_docs = results['documents'][0]

        if use_streaming:
            logger.info("Using streaming mode...")
            for streamed_text, structured_output in stream_generate_answer(
                query, context_docs, llm=llm
            ):
                if streamed_text:
                    print(streamed_text, end="", flush=True)
                if structured_output:
                    # Print structured output at the end
                    print("\n")
                    logger.info("Structured Output:")
                    print(structured_output.to_json())
        else:
            structured_output = generate_answer(query, context_docs, llm=llm)
            logger.info(f"  {structured_output.answer}")
            logger.info("Structured Output:")
            print(structured_output.to_json())

    logger.info("")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Parse arguments
    # Usage: python RAGPDFDemo.py [pdf_path] [--llm minimax|deepseek] [--stream]
    PDF_PATH = "AI_Study_Material.pdf"
    llm = DEFAULT_LLM
    use_streaming = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--llm" and i + 1 < len(sys.argv):
            llm = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--stream":
            use_streaming = True
            i += 1
        else:
            PDF_PATH = sys.argv[i]
            i += 1

    logger.info("=" * 80)
    logger.info("RAG PDF Demo v1.0 - Production Ready")
    logger.info("=" * 80)
    logger.info(f"LLM: {llm}, Streaming: {use_streaming}")

    # Run pipeline with semantic chunking (best for preserving meaning)
    collection = run_rag_pipeline(
        pdf_path=PDF_PATH,
        chunking_strategy="semantic",
        max_sentences=2
    )

    if collection:
        # Demo retrieval with default LLM (MiniMax)
        logger.info("=" * 80)
        logger.info("RETRIEVAL DEMO (LLM: " + ("MiniMax" if llm == "minimax" else "DeepSeek") + ")")
        logger.info("=" * 80)

        # To use DeepSeek: pass llm="deepseek" to demo_retrieval
        # demo_retrieval(collection, query, llm="deepseek")
        demo_retrieval(collection, "What is machine learning?", llm=llm, use_streaming=use_streaming)
        demo_retrieval(collection, "Explain deep learning neural networks", llm=llm, use_streaming=use_streaming)
        demo_retrieval(collection, "Natural language processing applications", llm=llm, use_streaming=use_streaming)
