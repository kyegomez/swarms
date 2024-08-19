import asyncio
import json
import os
import glob
from datetime import datetime
from typing import Dict, Literal
from chromadb.config import Settings
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from swarms.server.async_parent_document_retriever import AsyncParentDocumentRetriever

store_type = "local"  # "redis" or "local"

class VectorStorage:
    def __init__(self, directoryOrUrl, useGPU=False):
        self.embeddings = HuggingFaceBgeEmbeddings(
            cache_folder="./.embeddings",
            model_name="BAAI/bge-large-en",
            model_kwargs={"device": "cuda" if useGPU else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: ",
        )
        self.directoryOrUrl = directoryOrUrl
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        if store_type == "redis":
            from langchain.storage import RedisStore
            from langchain.utilities.redis import get_client

            username = r"username"
            password = r"password"
            client = get_client(
                redis_url=f"redis://{username}:{password}@redis-10854.c282.east-us-mz.azure.cloud.redislabs.com:10854"
            )
            self.store = RedisStore(client=client)
        else:
            self.store = LocalFileStore(root_path="./.parent_documents")
        self.settings = Settings(
            persist_directory="./.chroma_db",
            is_persistent=True,
            anonymized_telemetry=False,
        )
        # create a new vectorstore or get an existing one, with default collection
        self.vectorstore = self.getVectorStore()
        self.client = self.vectorstore._client
        self.retrievers: Dict[str, BaseRetriever] = {}
        # default retriever for when no collection title is specified
        self.retrievers[
            str(self.vectorstore._LANGCHAIN_DEFAULT_COLLECTION_NAME)
        ] = self.vectorstore.as_retriever()

    async def initRetrievers(self, directories: list[str] | None = None):
        start_time = datetime.now()
        print(f"Start vectorstore initialization time: {start_time}")

        # for each subdirectory in the directory, create a new collection if it doesn't exist
        dirs = directories or os.listdir(self.directoryOrUrl)
        # make sure the subdir is not a file on MacOS (which has a hidden .DS_Store file)
        dirs = [
            subdir
            for subdir in dirs
            if not os.path.isfile(f"{self.directoryOrUrl}/{subdir}")
        ]
        print(f"{len(dirs)} subdirectories to load: {dirs}")

        self.retrievers[self.directoryOrUrl] = await self.initRetriever(self.directoryOrUrl)
        
        end_time = datetime.now()
        print("Vectorstore initialization complete.")
        print(f"Vectorstore initialization end time: {end_time}")
        print(f"Total time taken: {end_time - start_time}")

        return self.retrievers

    async def initRetriever(self, subdir: str) -> BaseRetriever:
        # Ensure only one process/thread is executing this method at a time
        lock = asyncio.Lock()
        async with lock:
            subdir_start_time = datetime.now()
            print(f"Start {subdir} processing time: {subdir_start_time}")

            # get all existing collections
            collections = self.client.list_collections()
            print(f"Existing collections: {collections}")

            # Initialize an empty list to hold the documents
            documents = []
            # Define the maximum number of files to load at a time
            max_files = 1000

            # Load existing metadata
            metadata_file = f"{self.directoryOrUrl}/metadata.json"
            metadata = {"processDate": str(datetime.now()), "processed_files": []}
            processed_files = set()  # Track processed files
            if os.path.isfile(metadata_file):
                with open(metadata_file, "r") as metadataFile:
                    metadata = dict[str, str](json.load(metadataFile))
                    processed_files = {entry["file"] for entry in metadata.get("processed_files", [])}

            # Get a list of all files in the directory and exclude processed files
            all_files = [
                file for file in glob.glob(f"{self.directoryOrUrl}/**/*.md", recursive=True)
                if file not in processed_files
            ]

            print(f"Loading {len(all_files)} documents for title version {subdir}.")
            # Load files in chunks of max_files
            for i in range(0, len(all_files), max_files):
                chunksStartTime = datetime.now()
                chunk_files = all_files[i : i + max_files]
                for file in chunk_files:
                    loader = UnstructuredMarkdownLoader(
                        file,
                        mode="single",
                        strategy="fast"
                    )
                    print(f"Loaded {file} in {subdir} ...")
                    documents.extend(loader.load())

                    # Record the file as processed in metadata
                    metadata["processed_files"].append({
                        "file": file,
                        "processed_at": str(datetime.now())
                    })

                    print(f"Creating new collection for {self.directoryOrUrl}...")
                    # Create or get the collection
                    collection = self.client.create_collection(
                        name=self.directoryOrUrl,
                        get_or_create=True,
                        metadata={"processDate": metadata["processDate"]},
                    )

                    # Reload vectorstore based on collection
                    vectorstore = self.getVectorStore(collection_name=self.directoryOrUrl)

                    # Create a new parent document retriever
                    retriever = AsyncParentDocumentRetriever(
                        docstore=self.store,
                        vectorstore=vectorstore,
                        child_splitter=self.child_splitter,
                        parent_splitter=self.parent_splitter,
                    )   

                    # force reload of collection to make sure we don't have the default langchain collection
                    collection = self.client.get_collection(name=self.directoryOrUrl)
                    vectorstore = self.getVectorStore(collection_name=self.directoryOrUrl)

                    # Add documents to the collection and docstore
                    print(f"Adding {len(documents)} documents to collection...")
                    add_docs_start_time = datetime.now()
                    await retriever.aadd_documents(
                        documents=documents, add_to_docstore=True
                    )
                    add_docs_end_time = datetime.now()
                    print(
                        f"Adding {len(documents)} documents to collection took: {add_docs_end_time - add_docs_start_time}"
                    )

                    documents = []  # clear documents list for next chunk

                    # Save metadata to the metadata.json file
                    with open(metadata_file, "w") as metadataFile:
                        json.dump(metadata, metadataFile, indent=4)

                print(f"Loaded {len(documents)} documents for directory '{subdir}'.")
                chunksEndTime = datetime.now()
                print(
                    f"{max_files} markdown file chunks processing time: {chunksEndTime - chunksStartTime}"
                )

            subdir_end_time = datetime.now()
            print(f"Subdir {subdir} processing end time: {subdir_end_time}")
            print(f"Time taken: {subdir_end_time - subdir_start_time}")

            # Reload vectorstore based on collection to pass to parent doc retriever
            # collection = self.client.get_collection(name=self.directoryOrUrl)
            vectorstore = self.getVectorStore()
            retriever = AsyncParentDocumentRetriever(
                docstore=self.store,
                vectorstore=vectorstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )
            return retriever

    def getVectorStore(self, collection_name: str | None = None) -> Chroma:
        if collection_name is None or "" or "None" :
            collection_name = "langchain"
        print("collection_name: " + collection_name)
        vectorstore = Chroma(
            client_settings=self.settings,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        return vectorstore

    def list_collections(self):
        vectorstore = Chroma(
            client_settings=self.settings, embedding_function=self.embeddings
        )
        return vectorstore._client.list_collections()

    async def getRetriever(self, collection_name: str | None = None):
        if self.retrievers is None:
            self.retrievers = await self.initRetrievers()

        if (
            collection_name is None
            or collection_name == ""
            or collection_name == "None"
        ):
            name = str(Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME)
        else:
            name = collection_name

        try:
            retriever = self.retrievers[name]
        except KeyError:
            print(f"Retriever for {name} not found, using default...")
            retriever = self.retrievers[Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME]

        return retriever
