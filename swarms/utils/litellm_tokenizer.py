import subprocess


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in the given text."""
    try:
        from litellm import encode
    except ImportError:
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "litellm"]
        )
        from litellm import encode

    return len(encode(model=model, text=text))


# if __name__ == "__main__":
#     print(count_tokens("Hello, how are you?"))
