"""
Chroma DB Demo: Store & query 100 embeddings
Demonstrates CRUD operations, indexing, similarity search, and metadata filtering
"""

import chromadb
import numpy as np

# Initialize Chroma client (persistent storage)
print("Initializing Chroma DB...")
client = chromadb.PersistentClient(path="./chroma_db")

# Clear any existing collection for fresh demo
try:
    client.delete_collection("documents")
except:
    pass

# Create collection - Chroma will auto-embed using its built-in model
collection = client.create_collection(
    name="documents",
    metadata={"description": "Demo collection with 100 document embeddings"}
)

# Sample documents across different categories
categories = {
    "technology": [
        "Python is a popular programming language for data science.",
        "Machine learning algorithms can identify patterns in data.",
        "Cloud computing enables scalable infrastructure services.",
        "Blockchain technology provides decentralized ledger systems.",
        "Artificial intelligence is transforming many industries.",
        "Cybersecurity protects systems from digital attacks.",
        "The internet connects billions of devices worldwide.",
        "Mobile apps have revolutionized consumer applications.",
        "Software development requires logical thinking skills.",
        "Database systems store and retrieve structured data.",
    ],
    "food": [
        "Pizza originated in Italy and is loved worldwide.",
        "Chinese cuisine includes diverse regional styles.",
        "Coffee contains caffeine which boosts energy.",
        "Vegetables provide essential vitamins and minerals.",
        "Fruits are natural sources of sugars and fiber.",
        "Baking requires precise measurements of ingredients.",
        "Tea is a popular beverage in many cultures.",
        "Chocolate is made from cocoa beans.",
        "Rice is a staple food for half the world.",
        "Spices add flavor and can have health benefits.",
    ],
    "sports": [
        "Football is the most popular sport globally.",
        "Basketball was invented by James Naismith in 1891.",
        "Tennis players compete on various court surfaces.",
        "Swimming is excellent for full body exercise.",
        "Running improves cardiovascular health.",
        "Yoga combines physical and mental practices.",
        "Cycling is both sport and transportation.",
        "Golf is played on expansive green courses.",
        "Baseball is called America's pastime.",
        "Soccer requires team coordination and stamina.",
    ],
    "science": [
        "Physics explains the fundamental laws of nature.",
        "Chemistry studies matter and its transformations.",
        "Biology explores living organisms and ecosystems.",
        "Astronomy examines celestial bodies in space.",
        "Geology studies Earth's structure and processes.",
        "Mathematics is the language of science.",
        "Evolution explains how species change over time.",
        "Climate science studies long-term weather patterns.",
        "Neuroscience examines the structure of the brain.",
        "Genetics determines inherited traits in organisms.",
    ],
    "travel": [
        "Paris is known for the Eiffel Tower and art museums.",
        "Tokyo combines ancient temples with modern technology.",
        "New York has iconic landmarks like Times Square.",
        "Beach resorts offer relaxation and water activities.",
        "Mountain climbing attracts adventure enthusiasts.",
        "Historical sites preserve cultural heritage.",
        "Cruises provide all-inclusive travel experiences.",
        "Passport requirements vary by country.",
        "Airlines offer different classes of service.",
        "Hotels range from budget to luxury accommodations.",
    ],
    "business": [
        "Marketing promotes products to target audiences.",
        "Finance manages money and investment decisions.",
        "Entrepreneurship involves starting new ventures.",
        "Leadership inspires teams to achieve goals.",
        "Economics studies production and consumption.",
        "Supply chain management optimizes product flow.",
        "Human resources handles employee relations.",
        "Sales drives revenue for organizations.",
        "Accounting tracks financial transactions.",
        "Strategy guides long-term business decisions.",
    ],
    "entertainment": [
        "Movies tell stories through visual media.",
        "Music evokes emotions through sound.",
        "Video games provide interactive entertainment.",
        "Theater combines acting, music, and dance.",
        "Television broadcasts programs to mass audiences.",
        "Comedy makes people laugh through humor.",
        "Documentaries present factual information.",
        "Streaming services deliver content online.",
        "Social media connects people digitally.",
        "Podcasts cover diverse topics in audio format.",
    ],
    "health": [
        "Exercise promotes physical and mental well-being.",
        "Sleep is essential for body recovery.",
        "Medication treats various medical conditions.",
        "Vaccines prevent infectious diseases.",
        "Mental health affects overall well-being.",
        "Nutrition influences energy and immunity.",
        "Hygiene prevents the spread of disease.",
        "Regular checkups detect health issues early.",
        "Stress management improves quality of life.",
        "Water is essential for human survival.",
    ],
    "nature": [
        "Forests provide habitat for countless species.",
        "Oceans cover over seventy percent of Earth.",
        "Mountains rise thousands of meters above sea level.",
        "Rivers flow from highlands to the sea.",
        "Deserts have extreme temperatures and low rainfall.",
        "Rainbows appear when sunlight refracts through water.",
        "Earthquakes result from tectonic plate movement.",
        "Volcanoes erupt with molten rock from below.",
        "Hurricanes are powerful tropical storms.",
        "Wildlife conservation protects endangered species.",
    ],
    "education": [
        "Schools provide formal education to children.",
        "Universities offer advanced degree programs.",
        "Online learning has grown significantly recently.",
        "Teachers facilitate student learning and growth.",
        "Libraries contain vast collections of books.",
        "Research advances knowledge in every field.",
        "Exams assess student understanding of material.",
        "Scholarships help students afford education.",
        "Study skills improve learning efficiency.",
        "Lifelong learning benefits personal development.",
    ]
}

# Flatten into list of documents with metadata
documents = []
metadatas = []
ids = []

doc_id = 0
for category, texts in categories.items():
    for text in texts:
        documents.append(text)
        metadatas.append({
            "category": category,
            "word_count": len(text.split())
        })
        ids.append(f"doc_{doc_id}")
        doc_id += 1

# Add all documents (100 total)
print(f"\nAdding {len(documents)} documents to Chroma DB...")

# Add to collection in batches - Chroma handles embedding automatically
batch_size = 20
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_metas = metadatas[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]

    collection.add(
        documents=batch_docs,
        metadatas=batch_metas,
        ids=batch_ids
    )
    print(f"  Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

print(f"\nTotal documents in collection: {collection.count()}")

# =============================================================================
# DEMO 1: Basic Similarity Search
# =============================================================================
print("\n" + "="*80)
print("DEMO 1: Basic Similarity Search")
print("="*80)

query_texts = [
    "machine learning and artificial intelligence",
    "delicious food and cooking",
    "team sports and competition"
]

for query in query_texts:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    print(f"\nQuery: '{query}'")
    print("-" * 50)
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        category = results['metadatas'][0][i]['category']
        similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
        print(f"  {i+1}. [{category}] {results['documents'][0][i][:50]}...")
        print(f"     Similarity: {similarity:.4f}")

# =============================================================================
# DEMO 2: Metadata Filtering
# =============================================================================
print("\n" + "="*80)
print("DEMO 2: Metadata Filtering")
print("="*80)

# Filter by category
print("\nQuery: 'exercise and fitness' (only 'health' category)")
print("-" * 50)

results = collection.query(
    query_texts=["exercise and fitness"],
    n_results=5,
    where={"category": "health"}
)

for i in range(len(results['ids'][0])):
    category = results['metadatas'][0][i]['category']
    similarity = 1 - results['distances'][0][i]
    print(f"  {i+1}. [{category}] {results['documents'][0][i][:50]}...")
    print(f"     Similarity: {similarity:.4f}")

# Filter by word count
print("\nQuery: 'technology' (short documents, word_count < 10)")
print("-" * 50)

results = collection.query(
    query_texts=["technology and software"],
    n_results=5,
    where={"word_count": {"$lt": 10}}
)

for i in range(len(results['ids'][0])):
    word_count = results['metadatas'][0][i]['word_count']
    similarity = 1 - results['distances'][0][i]
    print(f"  {i+1}. [words: {word_count}] {results['documents'][0][i][:50]}...")
    print(f"     Similarity: {similarity:.4f}")

# =============================================================================
# DEMO 3: Combined Search + Filter
# =============================================================================
print("\n" + "="*80)
print("DEMO 3: Combined Semantic Search + Metadata Filter")
print("="*80)

print("\nQuery: 'learning' (only from technology OR science, max 3 results)")
print("-" * 50)

results = collection.query(
    query_texts=["learning"],
    n_results=3,
    where={"$or": [
        {"category": "technology"},
        {"category": "science"}
    ]}
)

for i in range(len(results['ids'][0])):
    category = results['metadatas'][0][i]['category']
    similarity = 1 - results['distances'][0][i]
    print(f"  {i+1}. [{category}] {results['documents'][0][i][:50]}...")
    print(f"     Similarity: {similarity:.4f}")

# =============================================================================
# DEMO 4: Get by ID
# =============================================================================
print("\n" + "="*80)
print("DEMO 4: Get Document by ID")
print("="*80)

results = collection.get(ids=["doc_0", "doc_25", "doc_50"])

print("\nFetching specific documents by ID:")
for i, doc_id in enumerate(results['ids']):
    print(f"\n  ID: {doc_id}")
    print(f"  Category: {results['metadatas'][i]['category']}")
    print(f"  Text: {results['documents'][i]}")

# =============================================================================
# DEMO 5: Update and Delete
# =============================================================================
print("\n" + "="*80)
print("DEMO 5: Update and Delete Operations")
print("="*80)

# Update a document
print("\nUpdating doc_0...")
collection.update(
    ids=["doc_0"],
    documents=["UPDATED: Python is now the world's most popular programming language!"],
    metadatas=[{"category": "technology", "word_count": 9, "updated": True}]
)

result = collection.get(ids=["doc_0"])
print(f"  Updated text: {result['documents'][0]}")
print(f"  Updated metadata: {result['metadatas'][0]}")

# Add a new document (to demonstrate delete effect)
print("\nAdding a new test document...")
test_id = "test_doc_delete"

collection.add(
    documents=["This is a test document for deletion"],
    metadatas=[{"category": "test", "word_count": 6}],
    ids=[test_id]
)

print(f"  Collection count before delete: {collection.count()}")

# Delete the test document
collection.delete(ids=[test_id])
print(f"  Collection count after delete: {collection.count()}")

print("\n" + "="*80)
print("Demo Complete!")
print("="*80)
print(f"\nCollection location: ./chroma_db")
print(f"Total documents: {collection.count()}")
