import pytest
import torch
from swarms.models.jina_embeds import JinaEmbeddings


@pytest.fixture
def model():
    return JinaEmbeddings("bert-base-uncased", verbose=True)


def test_initialization(model):
    assert isinstance(model, JinaEmbeddings)
    assert model.device in ["cuda", "cpu"]
    assert model.max_length == 500
    assert model.verbose is True


def test_run_sync(model):
    task = "Encode this text"
    result = model.run(task)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (model.max_length,)


def test_run_async(model):
    task = "Encode this text"
    result = model.run_async(task)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (model.max_length,)


def test_save_model(tmp_path, model):
    model_path = tmp_path / "model"
    model.save_model(model_path)
    assert (model_path / "config.json").is_file()
    assert (model_path / "pytorch_model.bin").is_file()
    assert (model_path / "vocab.txt").is_file()


def test_gpu_available(model):
    gpu_status = model.gpu_available()
    if torch.cuda.is_available():
        assert gpu_status is True
    else:
        assert gpu_status is False


def test_memory_consumption(model):
    memory_stats = model.memory_consumption()
    if torch.cuda.is_available():
        assert "allocated" in memory_stats
        assert "reserved" in memory_stats
    else:
        assert "error" in memory_stats


def test_cosine_similarity(model):
    task1 = "This is a sample text for testing."
    task2 = "Another sample text for testing."
    embeddings1 = model.run(task1)
    embeddings2 = model.run(task2)
    sim = model.cos_sim(embeddings1, embeddings2)
    assert isinstance(sim, torch.Tensor)
    assert sim.item() >= -1.0 and sim.item() <= 1.0


def test_failed_load_model(caplog):
    with pytest.raises(Exception):
        JinaEmbeddings("invalid_model")
    assert "Failed to load the model or the tokenizer" in caplog.text


def test_failed_generate_text(caplog, model):
    with pytest.raises(Exception):
        model.run("invalid_task")
    assert "Failed to generate the text" in caplog.text


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_change_device(model, device):
    model.device = device
    assert model.device == device
