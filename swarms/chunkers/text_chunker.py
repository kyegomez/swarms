from swarms.chunkers.base_chunker import BaseChunker
from swarms.chunkers.chunk_seperator import ChunkSeparator


class TextChunker(BaseChunker):
    DEFAULT_SEPARATORS = [
        ChunkSeparator("\n\n"),
        ChunkSeparator("\n"),
        ChunkSeparator(". "),
        ChunkSeparator("! "),
        ChunkSeparator("? "),
        ChunkSeparator(" "),
    ]
