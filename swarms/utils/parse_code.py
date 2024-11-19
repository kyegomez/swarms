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

    # Check if markdown_content is a string
    if not isinstance(markdown_content, str):
        raise TypeError("markdown_content must be a string")

    # Find all matches of the pattern
    matches = re.finditer(pattern, markdown_content, re.DOTALL)

    # Extract the content inside the backticks
    code_blocks = []
    for match in matches:
        code_block = match.group(1).strip()
        # Remove any leading or trailing whitespace from the code block
        code_block = code_block.strip()
        # Remove any empty lines from the code block
        code_block = "\n".join(
            [line for line in code_block.split("\n") if line.strip()]
        )
        code_blocks.append(code_block)

    # Concatenate all code blocks separated by newlines
    if code_blocks:
        return "\n\n".join(code_blocks)
    else:
        return ""


# example = """
# hello im an agent
# ```bash
# pip install swarms
# ```
# """

# print(extract_code_from_markdown(example))  # Output: { "type": "function", "function": { "name": "fetch_financial_news", "parameters": { "query": "Nvidia news", "num_articles": 5 } } }
