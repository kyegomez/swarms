from httpx import RequestError
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
class Qdrant:
    def __init__(self,api_key, host, port=6333, collection_name="qdrant", model_name="BAAI/bge-small-en-v1.5", https=True ):
        self.client = QdrantClient(url=host, port=port, api_key=api_key) #, port=port, api_key=api_key, https=False
        self.collection_name = collection_name
        self._load_embedding_model(model_name)
        self._setup_collection()

    def _load_embedding_model(self, model_name):
        # Load the embedding model
        self.model = SentenceTransformer(model_name)

    def _setup_collection(self):
        # Check if the collection already exists
        try:
            exists = self.client.get_collection(self.collection_name)
            return
        except Exception:
            # Collection does not exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=Distance.DOT),
            )
            print(f"Collection '{self.collection_name}' created.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def add_vectors(self, docs):
        # Add vectors with payloads to the collection
        points = []
        for i, doc in enumerate(docs):
            if doc.page_content:
                embedding = self.model.encode(doc.page_content, normalize_embeddings=True)
                points.append(PointStruct(id=i + 1, vector=embedding, payload={"content":doc.page_content}))
            else:
                print(f"Document at index {i} is missing 'text' or 'payload' key")

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )
        print(operation_info)
    def search_vectors(self, query, limit=3):
        query_vector= self.model.encode(query, normalize_embeddings=True)
        # Search for similar vectors
        search_result = self.client.search(
            collection_name=self.collection_name, 
            query_vector=query_vector, 
            limit=limit
        )
        return search_result



#TODO, use kwargs in constructor, have search result be text