<<<<<<< HEAD
from openai import OpenAI
<<<<<<< HEAD

client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
from dotenv import load_dotenv
from os import getenv
=======
>>>>>>> master
=======
import os
from openai import OpenAI
>>>>>>> master

client = OpenAI()


<<<<<<< HEAD
def get_ada_embeddings(text: str, model: str = "text-embedding-ada-002"):
=======
def get_ada_embeddings(
    text: str, model: str = "text-embedding-ada-002"
):
>>>>>>> master
    """
    Simple function to get embeddings from ada

    Usage:
    >>> get_ada_embeddings("Hello World")
    >>> get_ada_embeddings("Hello World", model="text-embedding-ada-001")

    """
<<<<<<< HEAD
<<<<<<< HEAD
    

    text = text.replace("\n", " ")

    return client.embeddings.create(input=[text],
    model=model)["data"][
        0
    ]["embedding"]
=======

    text = text.replace("\n", " ")

    return client.embeddings.create(input=[text], model=model)["data"][0]["embedding"]
>>>>>>> master
=======

    text = text.replace("\n", " ")

    return client.embeddings.create(input=[text], model=model)[
        "data"
    ][0]["embedding"]
>>>>>>> master
