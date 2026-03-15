"""
Embeddings Demo: Embed 10 sentences and compute similarity matrix
Uses spaCy with word vectors (no external API needed)
"""

import spacy
import numpy as np

# Load English model with word vectors
print("Loading spaCy model (with word vectors)...")
nlp = spacy.load("en_core_web_md")  # medium model has word vectors

# 10 diverse sentences
sentences = [
    "The cat sat on the mat.",
    "A feline was resting on the floor covering.",
    "Dogs are loyal companions.",
    "I love eating pizza with friends.",
    "Pizza is my favorite food.",
    "The weather is sunny today.",
    "It's raining heavily outside.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "The stock market crashed yesterday."
]

print("Generating embeddings for 10 sentences...\n")

# Generate embeddings using spaCy
embeddings = []
for sentence in sentences:
    doc = nlp(sentence)
    # Use sentence vector (average of word vectors)
    embeddings.append(doc.vector)

embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Each sentence is represented as a {embeddings.shape[1]}-dimensional vector\n")

# Compute cosine similarity matrix
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Build similarity matrix
n = len(sentences)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

# Print similarity matrix
print("=" * 100)
print("COSINE SIMILARITY MATRIX")
print("=" * 100)

# Print column headers (shortened)
print("\n     ", end="")
for idx in range(n):
    print(f"  S{idx:02d}", end="")
print()

# Print matrix
for i in range(n):
    print(f"S{i:02d}", end="")
    for j in range(n):
        print(f" {similarity_matrix[i][j]:.3f}", end="")
    print()

# Show most similar sentence pairs (excluding self)
print("\n" + "=" * 100)
print("TOP 5 MOST SIMILAR SENTENCE PAIRS")
print("=" * 100)

pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((i, j, similarity_matrix[i][j]))

# Sort by similarity (descending)
pairs.sort(key=lambda x: x[2], reverse=True)

for rank, (i, j, sim) in enumerate(pairs[:5], 1):
    print(f"\n{rank}. Similarity: {sim:.4f}")
    print(f"   Sentence {i}: {sentences[i]}")
    print(f"   Sentence {j}: {sentences[j]}")

# Show most dissimilar pairs
print("\n" + "=" * 100)
print("TOP 5 MOST DISSIMILAR SENTENCE PAIRS")
print("=" * 100)

for rank, (i, j, sim) in enumerate(pairs[-5:], 1):
    print(f"\n{rank}. Similarity: {sim:.4f}")
    print(f"   Sentence {i}: {sentences[i]}")
    print(f"   Sentence {j}: {sentences[j]}")
