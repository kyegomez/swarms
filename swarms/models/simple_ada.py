import os
from openai import OpenAI

client = OpenAI()


def get_ada_embeddings(
    text: str, model: str = "text-embedding-ada-002"
):
    """
    Simple function to get embeddings from ada

    Usage:
    >>> get_ada_embeddings("Hello World")
    >>> get_ada_embeddings("Hello World", model="text-embedding-ada-001")

    """

    text = text.replace("\n", " ")

    return client.embeddings.create(input=[text], model=model)[
        "data"
    ][0]["embedding"]
