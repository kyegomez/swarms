from swarms.memory import ChromaDB

# Initialize the memory
chroma = ChromaDB(
    metric="cosine",
    limit_tokens=1000,
    verbose=True,
)

# Add text
text = "This is a test"
chroma.add(text)

# Search for similar text
similar_text = chroma.query(text)
