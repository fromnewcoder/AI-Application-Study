# Vector Databases & Chroma DB

## What are Vector Databases?

Vector databases are specialized databases designed to store and query **high-dimensional vector embeddings** efficiently. They're optimized for:

- **Similarity search**: Find nearest neighbors in N-dimensional space
- **Semantic retrieval**: Query by meaning, not exact matches
- **Scalability**: Handle millions/billions of vectors
- **Fast queries**: Use Approximate Nearest Neighbor (ANN) algorithms

### Use Cases
- **RAG (Retrieval-Augmented Generation)**: Store document embeddings, retrieve relevant context
- **Recommendation systems**: Find similar items based on embedding similarity
- **Semantic search**: Search by meaning, not keywords
- **Image/video similarity**: Find visually similar media

## Chroma DB Overview

Chroma is an open-source vector database designed for AI applications.

### Key Features
- **Simple API**: Easy to use, minimal setup
- **Persistent storage**: Data saved to disk
- **Built-in embeddings**: Automatic embedding computation
- **Metadata filtering**: Filter by metadata before similarity search
- **Client-server mode**: Run as a server for production

### Installation
```bash
pip install chromadb
```

## CRUD Operations

### Create (Add embeddings)
```python
import chromadb

# Create client (in-memory or persistent)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = client.create_collection("my_collection")

# Add embeddings with metadata
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["Document 1 text", "Document 2 text"],
    metadatas=[{"source": "web"}, {"source": "pdf"}]
)
```

### Read (Query embeddings)
```python
# Query by embedding vector
results = collection.query(
    query_embeddings=[[1.0, 2.0, 3.0]],
    n_results=2
)

# Get by ID
item = collection.get(ids=["doc1"])
```

### Update
```python
# Update existing document
collection.update(
    ids=["doc1"],
    documents=["Updated text"],
    metadatas=[{"source": "updated"}]
)
```

### Delete
```python
collection.delete(ids=["doc1", "doc2"])
collection.delete(where={"source": "web"})  # Delete by filter
```

## Indexing

Chroma uses **Approximate Nearest Neighbor (ANN)** algorithms for fast similarity search:

| Index Type | Description | Speed | Accuracy |
|------------|-------------|-------|----------|
| HNSW | Hierarchical Navigable Small World | Very Fast | High |
| IVF | Inverted File Index | Fast | Medium |
| Flat | Brute force | Slow | Perfect |

### Default: HNSW
Chroma uses HNSW by default - excellent speed/accuracy balance.

```python
# Chroma automatically creates HNSW index
collection = client.create_collection(
    "my_collection",
    hnsw_space="cosine"  # or "l2", "ip" (inner product)
)
```

## Similarity Search

### Query Types

1. **By embedding vector** (most common)
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=10
)
```

2. **By text** (Chroma auto-embeds)
```python
results = collection.query(
    query_texts=["search term"],
    n_results=10
)
```

### Distance Metrics

| Metric | Best For | Range |
|--------|----------|-------|
| Cosine | Semantic similarity | 0-1 (higher = similar) |
| L2 (Euclidean) | Geometric distance | 0-∞ (lower = similar) |
| IP (Inner Product) | Unnormalized similarity | -1 to 1 |

```python
# Set distance metric when creating collection
collection = client.create_collection(
    "my_collection",
    hnsw_space="cosine"  # recommended for semantic search
)
```

## Metadata Filtering

Filter results before similarity search for more precise results.

### Basic Filters
```python
# Filter by metadata
results = collection.query(
    query_texts=["search term"],
    n_results=10,
    where={"source": "pdf"},  # Equal filter
    # where={"year": {"$gte": 2020}}  # Greater than
)
```

### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"source": {"$eq": "web"}}` |
| `$ne` | Not equals | `{"active": {"$ne": false}}` |
| `$gt`, `$gte` | Greater than | `{"year": {"$gt": 2020}}` |
| `$lt`, `$lte` | Less than | `{"score": {"$lt": 0.5}}` |
| `$in` | In array | `{"category": {"$in": ["tech", "science"]}}` |
| `$exists` | Field exists | `{"author": {"$exists": true}}` |

### Combined: Semantic Search + Metadata Filter
```python
results = collection.query(
    query_texts=["machine learning"],
    n_results=5,
    where={
        "source": "web",
        "year": {"$gte": 2023}
    }
)
```

## Best Practices

1. **Choose right distance metric**: Cosine for semantic, L2 for geometric
2. **Filter early**: Use metadata filters to reduce search space
3. **Batch operations**: Add many items at once for performance
4. **Index tuning**: Adjust `hnsw_ef` and `hnsw_construction_ef` for speed/accuracy trade-off
5. **Dimension matching**: Ensure embedding dimensions match your model output
