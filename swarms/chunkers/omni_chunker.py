"""
Omni Chunker is a chunker that chunks all files into select chunks of size x strings

Usage:
--------------
from swarms.chunkers.omni_chunker import OmniChunker

# Example
pdf = "swarmdeck.pdf"
chunker = OmniChunker(chunk_size=1000, beautify=True)
chunks = chunker(pdf)
print(chunks)


"""
from dataclasses import dataclass
from typing import List, Optional, Callable
from termcolor import colored
import os
import sys


@dataclass
class OmniChunker:
    """ """

    chunk_size: int = 1000
    beautify: bool = False
    use_tokenizer: bool = False
    tokenizer: Optional[Callable[[str], List[str]]] = None

    def __call__(self, file_path: str) -> List[str]:
        """
        Chunk the given file into parts of size `chunk_size`.

        Args:
            file_path (str): The path to the file to chunk.

        Returns:
            List[str]: A list of string chunks from the file.
        """
        if not os.path.isfile(file_path):
            print(colored("The file does not exist.", "red"))
            return []

        file_extension = os.path.splitext(file_path)[1]
        try:
            with open(file_path, "rb") as file:
                content = file.read()
                # Decode content based on MIME type or file extension
                decoded_content = self.decode_content(content, file_extension)
                chunks = self.chunk_content(decoded_content)
                return chunks

        except Exception as e:
            print(colored(f"Error reading file: {e}", "red"))
            return []

    def decode_content(self, content: bytes, file_extension: str) -> str:
        """
        Decode the content of the file based on its MIME type or file extension.

        Args:
            content (bytes): The content of the file.
            file_extension (str): The file extension of the file.

        Returns:
            str: The decoded content of the file.
        """
        # Add logic to handle different file types based on the extension
        # For simplicity, this example assumes text files encoded in utf-8
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as e:
            print(
                colored(
                    f"Could not decode file with extension {file_extension}: {e}",
                    "yellow",
                ))
            return ""

    def chunk_content(self, content: str) -> List[str]:
        """
        Split the content into chunks of size `chunk_size`.

        Args:
            content (str): The content to chunk.

        Returns:
            List[str]: The list of chunks.
        """
        return [
            content[i:i + self.chunk_size]
            for i in range(0, len(content), self.chunk_size)
        ]

    def __str__(self):
        return f"OmniChunker(chunk_size={self.chunk_size}, beautify={self.beautify})"

    def metrics(self):
        return {
            "chunk_size": self.chunk_size,
            "beautify": self.beautify,
        }

    def print_dashboard(self):
        print(
            colored(
                f"""
            Omni Chunker
            ------------
            {self.metrics()}
            """,
                "cyan",
            ))
