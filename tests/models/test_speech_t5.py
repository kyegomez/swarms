import os

import pytest
import torch

from swarms.models.speecht5 import SpeechT5


# Create fixtures if needed
@pytest.fixture
def speecht5_model():
    return SpeechT5()


# Test cases for the SpeechT5 class


def test_speecht5_init(speecht5_model):
    assert isinstance(
        speecht5_model.processor, SpeechT5.processor.__class__
    )
    assert isinstance(speecht5_model.model, SpeechT5.model.__class__)
    assert isinstance(
        speecht5_model.vocoder, SpeechT5.vocoder.__class__
    )
    assert isinstance(
        speecht5_model.embeddings_dataset, torch.utils.data.Dataset
    )


def test_speecht5_call(speecht5_model):
    text = "Hello, how are you?"
    speech = speecht5_model(text)
    assert isinstance(speech, torch.Tensor)


def test_speecht5_save_speech(speecht5_model):
    text = "Hello, how are you?"
    speech = speecht5_model(text)
    filename = "test_speech.wav"
    speecht5_model.save_speech(speech, filename)
    assert os.path.isfile(filename)
    os.remove(filename)


def test_speecht5_set_model(speecht5_model):
    old_model_name = speecht5_model.model_name
    new_model_name = "facebook/speecht5-tts"
    speecht5_model.set_model(new_model_name)
    assert speecht5_model.model_name == new_model_name
    assert speecht5_model.processor.model_name == new_model_name
    assert (
        speecht5_model.model.config.model_name_or_path
        == new_model_name
    )
    speecht5_model.set_model(old_model_name)  # Restore original model


def test_speecht5_set_vocoder(speecht5_model):
    old_vocoder_name = speecht5_model.vocoder_name
    new_vocoder_name = "facebook/speecht5-hifigan"
    speecht5_model.set_vocoder(new_vocoder_name)
    assert speecht5_model.vocoder_name == new_vocoder_name
    assert (
        speecht5_model.vocoder.config.model_name_or_path
        == new_vocoder_name
    )
    speecht5_model.set_vocoder(
        old_vocoder_name
    )  # Restore original vocoder


def test_speecht5_set_embeddings_dataset(speecht5_model):
    old_dataset_name = speecht5_model.dataset_name
    new_dataset_name = "Matthijs/cmu-arctic-xvectors-test"
    speecht5_model.set_embeddings_dataset(new_dataset_name)
    assert speecht5_model.dataset_name == new_dataset_name
    assert isinstance(
        speecht5_model.embeddings_dataset, torch.utils.data.Dataset
    )
    speecht5_model.set_embeddings_dataset(
        old_dataset_name
    )  # Restore original dataset


def test_speecht5_get_sampling_rate(speecht5_model):
    sampling_rate = speecht5_model.get_sampling_rate()
    assert sampling_rate == 16000


def test_speecht5_print_model_details(speecht5_model, capsys):
    speecht5_model.print_model_details()
    captured = capsys.readouterr()
    assert "Model Name: " in captured.out
    assert "Vocoder Name: " in captured.out


def test_speecht5_quick_synthesize(speecht5_model):
    text = "Hello, how are you?"
    speech = speecht5_model.quick_synthesize(text)
    assert isinstance(speech, list)
    assert isinstance(speech[0], dict)
    assert "audio" in speech[0]


def test_speecht5_change_dataset_split(speecht5_model):
    split = "test"
    speecht5_model.change_dataset_split(split)
    assert speecht5_model.embeddings_dataset.split == split


def test_speecht5_load_custom_embedding(speecht5_model):
    xvector = [0.1, 0.2, 0.3, 0.4, 0.5]
    embedding = speecht5_model.load_custom_embedding(xvector)
    assert torch.all(
        torch.eq(embedding, torch.tensor(xvector).unsqueeze(0))
    )


def test_speecht5_with_different_speakers(speecht5_model):
    text = "Hello, how are you?"
    speakers = [7306, 5324, 1234]
    for speaker_id in speakers:
        speech = speecht5_model(text, speaker_id=speaker_id)
        assert isinstance(speech, torch.Tensor)


def test_speecht5_save_speech_with_different_extensions(
    speecht5_model,
):
    text = "Hello, how are you?"
    speech = speecht5_model(text)
    extensions = [".wav", ".flac"]
    for extension in extensions:
        filename = f"test_speech{extension}"
        speecht5_model.save_speech(speech, filename)
        assert os.path.isfile(filename)
        os.remove(filename)


def test_speecht5_invalid_speaker_id(speecht5_model):
    text = "Hello, how are you?"
    invalid_speaker_id = (
        9999  # Speaker ID that does not exist in the dataset
    )
    with pytest.raises(IndexError):
        speecht5_model(text, speaker_id=invalid_speaker_id)


def test_speecht5_invalid_save_path(speecht5_model):
    text = "Hello, how are you?"
    speech = speecht5_model(text)
    invalid_path = "/invalid_directory/test_speech.wav"
    with pytest.raises(FileNotFoundError):
        speecht5_model.save_speech(speech, invalid_path)


def test_speecht5_change_vocoder_model(speecht5_model):
    text = "Hello, how are you?"
    old_vocoder_name = speecht5_model.vocoder_name
    new_vocoder_name = "facebook/speecht5-hifigan-ljspeech"
    speecht5_model.set_vocoder(new_vocoder_name)
    speech = speecht5_model(text)
    assert isinstance(speech, torch.Tensor)
    speecht5_model.set_vocoder(
        old_vocoder_name
    )  # Restore original vocoder
