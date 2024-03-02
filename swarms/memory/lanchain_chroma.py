import threading
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from swarms.models.openai_models import OpenAIChat


def synchronized_mem(method):
    """
    Decorator that synchronizes access to a method using a lock.

    Args:
        method: The method to be decorated.

    Returns:
        The decorated method.
    """

    def wrapper(self, *args, **kwargs):
        with self.lock:
            try:
                return method(self, *args, **kwargs)
            except Exception as e:
                print(f"Failed to execute {method.__name__}: {e}")

    return wrapper


class LangchainChromaVectorMemory:
    """
    A class representing a vector memory for storing and retrieving text entries.

    Attributes:
        loc (str): The location of the vector memory.
        chunk_size (int): The size of each text chunk.
        chunk_overlap_frac (float): The fraction of overlap between text chunks.
        embeddings (OpenAIEmbeddings): The embeddings used for text representation.
        count (int): The current count of text entries in the vector memory.
        lock (threading.Lock): A lock for thread safety.
        db (Chroma): The Chroma database for storing text entries.
        qa (RetrievalQA): The retrieval QA system for answering questions.

    Methods:
        __init__: Initializes the VectorMemory object.
        _init_db: Initializes the Chroma database.
        _init_retriever: Initializes the retrieval QA system.
        add_entry: Adds an entry to the vector memory.
        search_memory: Searches the vector memory for similar entries.
        ask_question: Asks a question to the vector memory.
    """

    def __init__(
        self,
        loc=None,
        chunk_size: int = 1000,
        chunk_overlap_frac: float = 0.1,
        *args,
        **kwargs,
    ):
        """
        Initializes the VectorMemory object.

        Args:
            loc (str): The location of the vector memory. If None, defaults to "./tmp/vector_memory".
            chunk_size (int): The size of each text chunk.
            chunk_overlap_frac (float): The fraction of overlap between text chunks.
        """
        if loc is None:
            loc = "./tmp/vector_memory"
        self.loc = Path(loc)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_size * chunk_overlap_frac
        self.embeddings = OpenAIEmbeddings()
        self.count = 0
        self.lock = threading.Lock()

        self.db = self._init_db()
        self.qa = self._init_retriever()

    def _init_db(self):
        """
        Initializes the Chroma database.

        Returns:
            Chroma: The initialized Chroma database.
        """
        texts = [
            "init"
        ]  # TODO find how to initialize Chroma without any text
        chroma_db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory=str(self.loc),
        )
        self.count = chroma_db._collection.count()
        return chroma_db

    def _init_retriever(self):
        """
        Initializes the retrieval QA system.

        Returns:
            RetrievalQA: The initialized retrieval QA system.
        """
        model = OpenAIChat(
            model_name="gpt-3.5-turbo",
        )
        qa_chain = load_qa_chain(model, chain_type="stuff")
        retriever = self.db.as_retriever(
            search_type="mmr", search_kwargs={"k": 10}
        )
        qa = RetrievalQA(
            combine_documents_chain=qa_chain, retriever=retriever
        )
        return qa

    @synchronized_mem
    def add(self, entry: str):
        """
        Add an entry to the internal memory.

        Args:
            entry (str): The entry to be added.

        Returns:
            bool: True if the entry was successfully added, False otherwise.
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
        )
        texts = text_splitter.split_text(entry)

        self.db.add_texts(texts)
        self.count += self.db._collection.count()
        self.db.persist()
        return True

    @synchronized_mem
    def search_memory(
        self, query: str, k=10, type="mmr", distance_threshold=0.5
    ):
        """
        Searching the vector memory for similar entries.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.
            type (str): The type of search to perform: "cos" or "mmr".
            distance_threshold (float): The similarity threshold to use for the search. Results with distance > similarity_threshold will be dropped.

        Returns:
            list[str]: A list of the top k results.
        """
        self.count = self.db._collection.count()
        if k > self.count:
            k = self.count - 1
        if k <= 0:
            return None

        if type == "mmr":
            texts = self.db.max_marginal_relevance_search(
                query=query, k=k, fetch_k=min(20, self.count)
            )
            texts = [text.page_content for text in texts]
        elif type == "cos":
            texts = self.db.similarity_search_with_score(
                query=query, k=k
            )
            texts = [
                text[0].page_content
                for text in texts
                if text[-1] < distance_threshold
            ]

        return texts

    @synchronized_mem
    def query(self, question: str):
        """
        Ask a question to the vector memory.

        Args:
            question (str): The question to ask.

        Returns:
            str: The answer to the question.
        """
        answer = self.qa.run(question)
        return answer
