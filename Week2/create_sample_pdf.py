"""Create a sample PDF for RAG demo."""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# Create PDF
pdf_path = "AI_Study_Material.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
styles = getSampleStyleSheet()
story = []

# Content
content = [
    ("Artificial Intelligence: A Comprehensive Overview", "Heading1"),
    ("Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.", "BodyText"),
    ("Machine Learning Fundamentals", "Heading1"),
    ("Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.", "BodyText"),
    ("Deep Learning Architecture", "Heading1"),
    ("Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs.", "BodyText"),
    ("Natural Language Processing", "Heading1"),
    ("Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of 'understanding' the contents of documents, including the contextual nuances of the language within them.", "BodyText"),
    ("Computer Vision Applications", "Heading1"),
    ("Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information.", "BodyText"),
    ("RAG: Retrieval Augmented Generation", "Heading1"),
    ("Retrieval Augmented Generation (RAG) is a technique for enhancing LLM output by retrieving relevant information from external knowledge bases. RAG combines the capabilities of large language models with information retrieval systems. The process involves: 1) Chunking documents into smaller segments, 2) Generating embeddings for each chunk, 3) Storing embeddings in a vector database, 4) Retrieving relevant chunks based on query similarity, 5) Augmenting the prompt with retrieved context.", "BodyText"),
    ("Vector Databases", "Heading1"),
    ("Vector databases are specialized databases that store and query high-dimensional vector embeddings. They enable semantic search by finding similar items based on vector similarity rather than exact matching. Popular vector databases include Chroma, Pinecone, Weaviate, and Milvus. These databases use approximate nearest neighbor (ANN) algorithms to efficiently search through millions of vectors.", "BodyText"),
    ("Text Chunking Strategies", "Heading1"),
    ("Text chunking is the process of dividing large documents into smaller, manageable pieces for embedding and retrieval. Common strategies include: Fixed-size chunking divides text into equal character lengths. Semantic chunking splits text at natural boundaries like sentences or paragraphs. Sliding window uses overlapping chunks to maintain context. Recursive chunking tries multiple separators to find natural break points.", "BodyText"),
]

# Build story
for text, style_name in content:
    p = Paragraph(text, styles[style_name])
    story.append(p)
    story.append(Spacer(1, 0.2 * inch))

# Build PDF
doc.build(story)
print(f"Created: {pdf_path}")
