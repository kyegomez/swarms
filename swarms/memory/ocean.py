#init ocean
# TODO upload ocean to pip and config it to the abstract class
import logging 
from typing import Union, List

import oceandb
from oceandb.utils.embedding_function import MultiModalEmbeddingFunction

class OceanDB:
    def __init__(self):
        try:
            self.client = oceandb.Client()
            print(self.client.heartbeat())
        except Exception as e:
            logging.error(f"Failed to initialize OceanDB client. Error: {e}")
    
    def create_collection(self, collection_name: str, modality: str):
        try:
            embedding_function = MultiModalEmbeddingFunction(modality=modality)
            collection = self.client.create_collection(collection_name, embedding_function=embedding_function)
            return collection
        except Exception as e:
            logging.error(f"Failed to create collection. Error {e}")

    def append_document(self, collection, document: str, id: str):
        try:
            return collection.add(documents=[document], ids[id])
        except Exception as e:
            logging.error(f"Faield to append document to the collection. Error {e}")
            raise
    
    def add_documents(self, collection, documents: List[str], ids: List[str]):
        try:
            return collection.add(documents=documents, ids=ids)
        except Exception as e:
            logging.error(f"Failed to add documents to collection. Error: {e}")
            raise

    def query(self, collection, query_texts: list[str], n_results: int):
        try:
            results = collection.query(query_texts=query_texts, n_results=n_results)
            return results
        except Exception as e:
            logging.error(f"Failed to query the collection. Error {e}")
            raise