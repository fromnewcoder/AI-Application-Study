# Embeddings Theory

## What are Embeddings?

Embeddings are **dense vector representations** of data (text, images, audio, etc.) that capture semantic meaning in a numerical format. They transform discrete, high-dimensional data into continuous, low-dimensional vectors where similar items are positioned close to each other in the vector space.

### Key Properties
- **Dense**: Most values are non-zero (vs. sparse one-hot encoding)
- **Continuous**: Values are floating-point numbers
- **Semantic**: Similar concepts have similar vectors
- **Fixed-size**: Each item gets a vector of consistent dimensionality

## Vector Space

A vector space is a mathematical space where points (vectors) have magnitude and direction. In embeddings:

- Each dimension represents a semantic feature
- The space is typically 256 to 4096 dimensions (for text embeddings)
- **Distance** between vectors = **similarity** between meanings

### Example 2D Visualization
```
                    dog
                     •
                cat •
                           car
              •apple
    banana •
```

## Cosine Similarity

Cosine similarity measures the angle between two vectors, ranging from -1 to 1:

- **1.0**: Identical direction (most similar)
- **0.0**: Orthogonal (unrelated)
- **-1.0**: Opposite direction (most dissimilar)

### Formula
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product of vectors
- `||A||` = magnitude (length) of vector A

## How Embedding Models Differ

| Model | Dimensions | Training Data | Strengths |
|-------|------------|---------------|-----------|
| OpenAI text-embedding-ada-002 | 1536 | Large web corpus | General purpose, high quality |
| OpenAI text-embedding-3-small | 1536 | Updated data | Faster, cheaper |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Sentence pairs | Fast, lightweight |
| Cohere embed-multilingual | 1024 | Multilingual | Multilingual support |
| DeepSeek | 1024 | Mixed | Cost-effective |

### Key Differences
1. **Dimensionality**: Higher isn't always better
2. **Training objective**: Contrastive vs. predictive
3. **Domain specialization**: General vs. domain-specific
4. **Language support**: English-only vs. multilingual

## Use Cases Overview

### 1. Semantic Search
Find documents by meaning, not just keywords.

### 2. Text Similarity
Compare how similar two pieces of text are.

### 3. Clustering
Group similar documents together automatically.

### 4. Recommendation Systems
Suggest items based on vector similarity.

### 5. Classification
Use embedding vectors as features for ML classifiers.

### 6. Anomaly Detection
Find outliers based on distance from cluster centers.

### 7. RAG (Retrieval-Augmented Generation)
Retrieve relevant context for LLM prompts.
