from swarms.memory import chroma

# loader = CSVLoader(
#     file_path="../document_parsing/aipg/aipg.csv",
#     encoding="utf-8-sig",
# )
# docs = loader.load()


# Initialize the Qdrant instance
# See qdrant documentation on how to run locally
qdrant_client = chroma.ChromaClient()

qdrant_client.add_vectors(["This is a document", "BONSAIIIIIII", "the walking dead"])

results = qdrant_client.search_vectors("zombie", limit=1)

print(results)

# qdrant_client.add_vectors(docs)
#
# # Perform a search
# search_query = "Who is jojo"
# search_results = qdrant_client.search_vectors(search_query)
# print("Search Results:")
# for result in search_results:
#     print(result)
