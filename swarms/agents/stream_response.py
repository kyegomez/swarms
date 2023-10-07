def stream(response):
    """
    Yield the response token by token (word by word) from llm
    """
    for token in response.split():
        yield token
