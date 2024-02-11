from swarms.models import JinaEmbeddings

model = JinaEmbeddings()

embeddings = model("Encode this text")

print(embeddings)
