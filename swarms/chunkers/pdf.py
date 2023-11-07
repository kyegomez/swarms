from swarms.chunkers.base import BaseChunker
from swarms.chunkers.chunk_seperator import ChunkSeparator


class PdfChunker(BaseChunker):
    DEFAULT_SEPARATORS = [
        ChunkSeparator("\n\n"),
        ChunkSeparator(". "),
        ChunkSeparator("! "),
        ChunkSeparator("? "),
        ChunkSeparator(" "),
    ]


# # Example
# pdf = "swarmdeck.pdf"
# chunker = PdfChunker()
# chunks = chunker.chunk(pdf)
# print(chunks)
