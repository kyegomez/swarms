# stock_price_server.py
from mcp.server.fastmcp import FastMCP
import os
from datetime import datetime

mcp = FastMCP("StockPrice")


@mcp.tool()
def create_markdown_file(filename: str) -> str:
    """
    Create a new markdown file with a basic structure.

    Args:
        filename (str): The name of the markdown file to create (without .md extension)

    Returns:
        str: A message indicating success or failure

    Example:
        >>> create_markdown_file('my_notes')
        'Created markdown file: my_notes.md'
    """
    try:
        if not filename:
            return "Please provide a valid filename"

        # Ensure filename ends with .md
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # Create basic markdown structure
        content = f"""# {filename.replace('.md', '')}
Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Content

"""

        with open(filename, "w") as f:
            f.write(content)

        return f"Created markdown file: {filename}"

    except Exception as e:
        return f"Error creating markdown file: {str(e)}"


@mcp.tool()
def write_to_markdown(filename: str, content: str) -> str:
    """
    Append content to an existing markdown file.

    Args:
        filename (str): The name of the markdown file (without .md extension)
        content (str): The content to append to the file

    Returns:
        str: A message indicating success or failure

    Example:
        >>> write_to_markdown('my_notes', 'This is a new note')
        'Content added to my_notes.md'
    """
    try:
        if not filename or not content:
            return "Please provide both filename and content"

        # Ensure filename ends with .md
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # Check if file exists
        if not os.path.exists(filename):
            return f"File {filename} does not exist. Please create it first using create_markdown_file"

        # Append content with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_content = f"\n### Entry - {timestamp}\n{content}\n"

        with open(filename, "a") as f:
            f.write(formatted_content)

        return f"Content added to {filename}"

    except Exception as e:
        return f"Error writing to markdown file: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="sse")
