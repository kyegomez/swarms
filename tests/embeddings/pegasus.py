import pytest
from unittest.mock import patch
from swarms.embeddings.pegasus import PegasusEmbedding


def test_init():
    with patch("your_module.Pegasus") as MockPegasus:
        embedder = PegasusEmbedding(modality="text")
        MockPegasus.assert_called_once()
        assert embedder.pegasus == MockPegasus.return_value


def test_init_exception():
    with patch("your_module.Pegasus", side_effect=Exception("Test exception")):
        with pytest.raises(Exception) as e:
            PegasusEmbedding(modality="text")
        assert str(e.value) == "Test exception"


def test_embed():
    with patch("your_module.Pegasus") as MockPegasus:
        embedder = PegasusEmbedding(modality="text")
        embedder.embed("Hello world")
        MockPegasus.return_value.embed.assert_called_once()


def test_embed_exception():
    with patch("your_module.Pegasus") as MockPegasus:
        MockPegasus.return_value.embed.side_effect = Exception("Test exception")
        embedder = PegasusEmbedding(modality="text")
        with pytest.raises(Exception) as e:
            embedder.embed("Hello world")
        assert str(e.value) == "Test exception"
