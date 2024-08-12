import re


def extract_code_from_markdown(markdown_content: str) -> str:
    """
    Extracts code blocks from a Markdown string and returns them as a single string.

    Args:
    - markdown_content (str): The Markdown content as a string.

    Returns:
    - str: A single string containing all the code blocks separated by newlines.
    """
    # Regular expression for fenced code blocks with optional language specifier
    pattern = r"```(?:\w+\n)?(.*?)```"

    # Find all matches of the pattern
    matches = re.finditer(pattern, markdown_content, re.DOTALL)

    # Extract the content inside the backticks
    code_blocks = [match.group(1).strip() for match in matches]

    # Concatenate all code blocks separated by newlines
    return "\n".join(code_blocks)


# example = """
# hello im an agent
# ```bash
# pip install swarms
# ```
# """

# print(extract_code_from_markdown(example))  # Output: { "type": "function", "function": { "name": "fetch_financial_news", "parameters": { "query": "Nvidia news", "num_articles": 5 } } }
