import time


def stream_response(response: str, delay: float = 0.1) -> None:
    """
    Streams the response token by token.

    Args:
        response (str): The response text to be streamed.
        delay (float, optional): Delay in seconds between printing each token. Default is 0.1 seconds.

    Raises:
        ValueError: If the response is not provided.
        Exception: For any errors encountered during the streaming process.

    Example:
        response = "This is a sample response from the API."
        stream_response(response)
    """
    # Check for required inputs
    if not response:
        raise ValueError("Response is required.")

    try:
        # Stream and print the response token by token
        for token in response.split():
            print(token, end=" ", flush=True)
            time.sleep(delay)
        print()  # Ensure a newline after streaming
    except Exception as e:
        print(f"An error occurred during streaming: {e}")


# # Example usage
# response = "This is a sample response from the API."
# stream_response(response)
