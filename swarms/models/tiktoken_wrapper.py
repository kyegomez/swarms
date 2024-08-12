import tiktoken


class TikTokenizer:
    def __init__(
        self,
        model_name: str = "o200k_base",
    ):
        """
        Initializes a TikTokenizer object.

        Args:
            model_name (str, optional): The name of the model to use for tokenization. Defaults to "gpt-4o".
        """
        try:
            self.model_name = model_name
            # self.tokenizer = tiktoken./(model_name)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize tokenizer with model '{model_name}': {str(e)}"
            )

    def count_tokens(self, string: str) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): The input text string.

        Returns:
            int: The number of tokens in the text string.
        """
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
