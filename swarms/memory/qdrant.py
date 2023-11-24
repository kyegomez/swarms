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

    def _load_embedding_model(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")

    def _setup_collection(self):
        try:
            exists = self.client.get_collection(self.collection_name)
            if exists:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=Distance.DOT),
            )
            print(f"Collection '{self.collection_name}' created.")

    def add_vectors(self, docs: List[dict]):
        points = []
        for i, doc in enumerate(docs):
            try:
                if 'page_content' in doc:
                    embedding = self.model.encode(doc['page_content'], normalize_embeddings=True)
                    points.append(PointStruct(id=i + 1, vector=embedding, payload={"content": doc['page_content']}))
                else:
                    print(f"Document at index {i} is missing 'page_content' key")
            except Exception as e:
                print(f"Error processing document at index {i}: {e}")

        try:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
            )
            return operation_info
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return None

    def search_vectors(self, query: str, limit: int = 3):
        try:
            query_vector = self.model.encode(query, normalize_embeddings=True)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return search_result
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return None
