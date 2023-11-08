import openai
from dotenv import load_dotenv
from os import getenv

load_dotenv()


def get_ada_embeddings(text: str, model: str = "text-embedding-ada-002"):
    """
    Simple function to get embeddings from ada

    Usage:
    >>> get_ada_embeddings("Hello World")
    >>> get_ada_embeddings("Hello World", model="text-embedding-ada-001")

    """
    openai.api_key = getenv("OPENAI_API_KEY")

    text = text.replace("\n", " ")

    return openai.Embedding.create(
        input=[text],
        model=model,
    )["data"][0]["embedding"]
