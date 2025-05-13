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


class LiteLLMTokenizer:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name

    def count(self, text):
        return count_tokens(text, model=self.model_name)


# if __name__ == "__main__":
#     print(count_tokens("Hello, how are you?"))
