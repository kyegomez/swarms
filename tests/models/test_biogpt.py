from unittest.mock import patch

# Import necessary modules
import pytest
import torch
from transformers import BioGptForCausalLM, BioGptTokenizer


# Fixture for BioGPT instance
@pytest.fixture
def biogpt_instance():
    from swarms.models import BioGPT

    return BioGPT()


# 36. Test if BioGPT provides a response for a simple biomedical question
def test_biomedical_response_1(biogpt_instance):
    question = "What are the functions of the mitochondria?"
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 37. Test for a genetics-based question
def test_genetics_response(biogpt_instance):
    question = "Can you explain the Mendelian inheritance?"
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 38. Test for a question about viruses
def test_virus_response(biogpt_instance):
    question = "How do RNA viruses replicate?"
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 39. Test for a cell biology related question
def test_cell_biology_response(biogpt_instance):
    question = "Describe the cell cycle and its phases."
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 40. Test for a question about protein structure
def test_protein_structure_response(biogpt_instance):
    question = (
        "What's the difference between alpha helix and beta sheet"
        " structures in proteins?"
    )
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 41. Test for a pharmacology question
def test_pharmacology_response(biogpt_instance):
    question = "How do beta blockers work?"
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 42. Test for an anatomy-based question
def test_anatomy_response(biogpt_instance):
    question = "Describe the structure of the human heart."
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 43. Test for a question about bioinformatics
def test_bioinformatics_response(biogpt_instance):
    question = "What is a BLAST search?"
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 44. Test for a neuroscience question
def test_neuroscience_response(biogpt_instance):
    question = "Explain the function of synapses in the nervous system."
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


# 45. Test for an immunology question
def test_immunology_response(biogpt_instance):
    question = "What is the role of T cells in the immune response?"
    response = biogpt_instance(question)
    assert response
    assert isinstance(response, str)


def test_init(bio_gpt):
    assert bio_gpt.model_name == "microsoft/biogpt"
    assert bio_gpt.max_length == 500
    assert bio_gpt.num_return_sequences == 5
    assert bio_gpt.do_sample is True
    assert bio_gpt.min_length == 100


def test_call(bio_gpt, monkeypatch):
    def mock_pipeline(*args, **kwargs):
        class MockGenerator:
            def __call__(self, text, **kwargs):
                return ["Generated text"]

        return MockGenerator()

    monkeypatch.setattr("transformers.pipeline", mock_pipeline)
    result = bio_gpt("Input text")
    assert result == ["Generated text"]


def test_get_features(bio_gpt):
    features = bio_gpt.get_features("Input text")
    assert "last_hidden_state" in features


def test_beam_search_decoding(bio_gpt):
    generated_text = bio_gpt.beam_search_decoding("Input text")
    assert isinstance(generated_text, str)


def test_set_pretrained_model(bio_gpt):
    bio_gpt.set_pretrained_model("new_model")
    assert bio_gpt.model_name == "new_model"


def test_get_config(bio_gpt):
    config = bio_gpt.get_config()
    assert "vocab_size" in config


def test_save_load_model(tmp_path, bio_gpt):
    bio_gpt.save_model(tmp_path)
    bio_gpt.load_from_path(tmp_path)
    assert bio_gpt.model_name == "microsoft/biogpt"


def test_print_model(capsys, bio_gpt):
    bio_gpt.print_model()
    captured = capsys.readouterr()
    assert "BioGptForCausalLM" in captured.out


# 26. Test if set_pretrained_model changes the model_name
def test_set_pretrained_model_name_change(biogpt_instance):
    biogpt_instance.set_pretrained_model("new_model_name")
    assert biogpt_instance.model_name == "new_model_name"


# 27. Test get_config return type
def test_get_config_return_type(biogpt_instance):
    config = biogpt_instance.get_config()
    assert isinstance(config, type(biogpt_instance.model.config))


# 28. Test saving model functionality by checking if files are created
@patch.object(BioGptForCausalLM, "save_pretrained")
@patch.object(BioGptTokenizer, "save_pretrained")
def test_save_model(mock_save_model, mock_save_tokenizer, biogpt_instance):
    path = "test_path"
    biogpt_instance.save_model(path)
    mock_save_model.assert_called_once_with(path)
    mock_save_tokenizer.assert_called_once_with(path)


# 29. Test loading model from path
@patch.object(BioGptForCausalLM, "from_pretrained")
@patch.object(BioGptTokenizer, "from_pretrained")
def test_load_from_path(
    mock_load_model, mock_load_tokenizer, biogpt_instance
):
    path = "test_path"
    biogpt_instance.load_from_path(path)
    mock_load_model.assert_called_once_with(path)
    mock_load_tokenizer.assert_called_once_with(path)


# 30. Test print_model doesn't raise any error
def test_print_model_metadata(biogpt_instance):
    try:
        biogpt_instance.print_model()
    except Exception as e:
        pytest.fail(f"print_model() raised an exception: {e}")


# 31. Test that beam_search_decoding uses the correct number of beams
@patch.object(BioGptForCausalLM, "generate")
def test_beam_search_decoding_num_beams(mock_generate, biogpt_instance):
    biogpt_instance.beam_search_decoding("test_sentence", num_beams=7)
    _, kwargs = mock_generate.call_args
    assert kwargs["num_beams"] == 7


# 32. Test if beam_search_decoding handles early_stopping
@patch.object(BioGptForCausalLM, "generate")
def test_beam_search_decoding_early_stopping(
    mock_generate, biogpt_instance
):
    biogpt_instance.beam_search_decoding(
        "test_sentence", early_stopping=False
    )
    _, kwargs = mock_generate.call_args
    assert kwargs["early_stopping"] is False


# 33. Test get_features return type
def test_get_features_return_type(biogpt_instance):
    result = biogpt_instance.get_features("This is a sample text.")
    assert isinstance(result, torch.nn.modules.module.Module)


# 34. Test if default model is set correctly during initialization
def test_default_model_name(biogpt_instance):
    assert biogpt_instance.model_name == "microsoft/biogpt"
