import inspect
import os
import re
import threading
from swarms import OpenAIChat
from scripts.auto_tests_docs.docs import TEST_WRITER_SOP_PROMPT

#########
from swarms.tokenizers.r_tokenizers import (
    SentencePieceTokenizer,
    HuggingFaceTokenizer,
    Tokenizer,
)
from swarms.tokenizers.base_tokenizer import BaseTokenizer
from swarms.tokenizers.openai_tokenizers import OpenAITokenizer
from swarms.tokenizers.anthropic_tokenizer import (
    AnthropicTokenizer,
)
from swarms.tokenizers.cohere_tokenizer import CohereTokenizer

########
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    model_name="gpt-4",
    openai_api_key=api_key,
    max_tokens=4000,
)


def extract_code_from_markdown(markdown_content: str):
    """
    Extracts code blocks from a Markdown string and returns them as a single string.

    Args:
    - markdown_content (str): The Markdown content as a string.

    Returns:
    - str: A single string containing all the code blocks separated by newlines.
    """
    # Regular expression for fenced code blocks
    pattern = r"```(?:\w+\n)?(.*?)```"
    matches = re.findall(pattern, markdown_content, re.DOTALL)

    # Concatenate all code blocks separated by newlines
    return "\n".join(code.strip() for code in matches)


def create_test(cls):
    """
    Process the documentation for a given class using OpenAI model and save it in a Python file.
    """
    doc = inspect.getdoc(cls)
    source = inspect.getsource(cls)
    input_content = (
        "Class Name:"
        f" {cls.__name__}\n\nDocumentation:\n{doc}\n\nSource"
        f" Code:\n{source}"
    )
    print(input_content)

    # Process with OpenAI model (assuming the model's __call__ method takes this input and returns processed content)
    processed_content = model(
        TEST_WRITER_SOP_PROMPT(
            input_content, "swarms", "swarms.tokenizers"
        )
    )
    processed_content = extract_code_from_markdown(processed_content)

    doc_content = f"# {cls.__name__}\n\n{processed_content}\n"

    # Create the directory if it doesn't exist
    dir_path = "tests/tokenizers"
    os.makedirs(dir_path, exist_ok=True)

    # Write the processed documentation to a Python file
    file_path = os.path.join(dir_path, f"{cls.__name__.lower()}.py")
    with open(file_path, "w") as file:
        file.write(doc_content)


def main():
    classes = [
        SentencePieceTokenizer,
        HuggingFaceTokenizer,
        Tokenizer,
        BaseTokenizer,
        OpenAITokenizer,
        AnthropicTokenizer,
        CohereTokenizer,
    ]
    threads = []
    for cls in classes:
        thread = threading.Thread(target=create_test, args=(cls,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Tests generated in 'tests/tokenizers' directory.")


if __name__ == "__main__":
    main()
