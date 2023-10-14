# This file contains the function that embeds the input into a vector
from chromadb import EmbeddingFunction


def openai_embed(self, input, api_key, model_name):
    openai = EmbeddingFunction.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=model_name
    )
    embedding = openai(input)
    return embedding
