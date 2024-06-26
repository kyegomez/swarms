import tiktoken


class TikTokenizer:
    def __init__(
        self,
        model_name: str = "gpt-4o",
    ):
        """
        Initializes a TikTokenizer object.

        Args:
            model_name (str, optional): The name of the model to use for tokenization. Defaults to "gpt-4o".
        """
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize tokenizer with model '{model_name}': {str(e)}"
            )

    def len(self, string: str) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): The input text string.

        Returns:
            int: The number of tokens in the text string.
        """
        try:
            num_tokens = len(self.tokenizer.encode(string))
            print(f"Number of tokens: {num_tokens}")
            return num_tokens
        except Exception as e:
            raise ValueError(f"Failed to tokenize string: {str(e)}")

    def count_tokens(self, string: str) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): The input text string.

        Returns:
            int: The number of tokens in the text string.
        """
        try:
            num_tokens = len(self.tokenizer.encode(string))
            print(f"Number of tokens: {num_tokens}")
            return num_tokens
        except Exception as e:
            raise ValueError(f"Failed to count tokens: {str(e)}")
