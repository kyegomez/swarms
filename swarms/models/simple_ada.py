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

<<<<<<< HEAD
    return client.embeddings.create(input=[text], model=model)[
        "data"
    ][0]["embedding"]
=======
    return client.embeddings.create(input=[text], model=model)["data"][0][
        "embedding"
    ]
>>>>>>> 49c7b97c (code quality fixes: line length = 80)
