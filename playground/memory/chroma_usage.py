from swarms.memory import chroma

chromadbcl = chroma.ChromaClient()

chromadbcl.add_vectors(["This is a document", "BONSAIIIIIII", "the walking dead"])

results = chromadbcl.search_vectors("zombie", limit=1)

print(results)

