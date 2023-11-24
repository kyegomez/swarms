import logging
from typing import List

import oceandb
from oceandb.utils.embedding_function import MultiModalEmbeddingFunction


class OceanDB:
    """
    A class to interact with OceanDB.

    ...

    Attributes
    ----------
    client : oceandb.Client
        a client to interact with OceanDB

    Methods
    -------
    create_collection(collection_name: str, modality: str):
        Creates a new collection in OceanDB.
    append_document(collection, document: str, id: str):
        Appends a document to a collection in OceanDB.
    add_documents(collection, documents: List[str], ids: List[str]):
        Adds multiple documents to a collection in OceanDB.
    query(collection, query_texts: list[str], n_results: int):
        Queries a collection in OceanDB.
    """

    def __init__(self, client: oceandb.Client = None):
        """
        Constructs all the necessary attributes for the OceanDB object.

        Parameters
        ----------
            client : oceandb.Client, optional
                a client to interact with OceanDB (default is None, which creates a new client)
        """
        try:
            self.client = client if client else oceandb.Client()
            print(self.client.heartbeat())
        except Exception as e:
            logging.error(f"Failed to initialize OceanDB client. Error: {e}")
            raise

    def create_collection(self, collection_name: str, modality: str):
        """
        Creates a new collection in OceanDB.

        Parameters
        ----------
            collection_name : str
                the name of the new collection
            modality : str
                the modality of the new collection

        Returns
        -------
            collection
                the created collection
        """
        try:
            embedding_function = MultiModalEmbeddingFunction(modality=modality)
            collection = self.client.create_collection(
                collection_name, embedding_function=embedding_function
            )
            return collection
        except Exception as e:
            logging.error(f"Failed to create collection. Error {e}")
            raise

    def append_document(self, collection, document: str, id: str):
        """
        Appends a document to a collection in OceanDB.

        Parameters
        ----------
            collection
                the collection to append the document to
            document : str
                the document to append
            id : str
                the id of the document

        Returns
        -------
            result
                the result of the append operation
        """
        try:
            return collection.add(documents=[document], ids=[id])
        except Exception as e:
            logging.error(
                f"Failed to append document to the collection. Error {e}"
            )
            raise

    def add_documents(self, collection, documents: List[str], ids: List[str]):
        """
        Adds multiple documents to a collection in OceanDB.

        Parameters
        ----------
            collection
                the collection to add the documents to
            documents : List[str]
                the documents to add
            ids : List[str]
                the ids of the documents

        Returns
        -------
            result
                the result of the add operation
        """
        try:
            return collection.add(documents=documents, ids=ids)
        except Exception as e:
            logging.error(f"Failed to add documents to collection. Error: {e}")
            raise

    def query(self, collection, query_texts: list[str], n_results: int):
        """
        Queries a collection in OceanDB.

        Parameters
        ----------
            collection
                the collection to query
            query_texts : list[str]
                the texts to query
            n_results : int
                the number of results to return

        Returns
        -------
            results
                the results of the query
        """
        try:
            results = collection.query(
                query_texts=query_texts, n_results=n_results
            )
            return results
        except Exception as e:
            logging.error(f"Failed to query the collection. Error {e}")
            raise


# Example
# ocean = OceanDB()
# collection = ocean.create_collection("test", "text")
# ocean.append_document(collection, "hello world", "1")
# ocean.add_documents(collection, ["hello world", "hello world"], ["2", "3"])
# results = ocean.query(collection, ["hello world"], 3)
# print(results)
