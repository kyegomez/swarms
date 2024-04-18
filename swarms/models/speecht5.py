"""
SpeechT5 (TTS task)
SpeechT5 model fine-tuned for speech synthesis (text-to-speech) on LibriTTS.

This model was introduced in SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

SpeechT5 was first released in this repository, original weights. The license used is MIT.

Model Description
Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.

Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

Developed by: Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.
Shared by [optional]: Matthijs Hollemans
Model type: text-to-speech
Language(s) (NLP): [More Information Needed]
License: MIT
Finetuned from model [optional]: [More Information Needed]
Model Sources [optional]
Repository: [https://github.com/microsoft/SpeechT5/]
Paper: [https://arxiv.org/pdf/2110.07205.pdf]
Blog Post: [https://huggingface.co/blog/speecht5]
Demo: [https://huggingface.co/spaces/Matthijs/speecht5-tts-demo]

"""

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    pipeline,
)


class SpeechT5:
    """
    SpeechT5Wrapper


    Args:
        model_name (str, optional): Model name or path. Defaults to "microsoft/speecht5_tts".
        vocoder_name (str, optional): Vocoder name or path. Defaults to "microsoft/speecht5_hifigan".
        dataset_name (str, optional): Dataset name or path. Defaults to "Matthijs/cmu-arctic-xvectors".

    Attributes:
        model_name (str): Model name or path.
        vocoder_name (str): Vocoder name or path.
        dataset_name (str): Dataset name or path.
        processor (SpeechT5Processor): Processor for the SpeechT5 model.
        model (SpeechT5ForTextToSpeech): SpeechT5 model.
        vocoder (SpeechT5HifiGan): SpeechT5 vocoder.
        embeddings_dataset (datasets.Dataset): Dataset containing speaker embeddings.

    Methods
        __call__: Synthesize speech from text.
        save_speech: Save speech to a file.
        set_model: Change the model.
        set_vocoder: Change the vocoder.
        set_embeddings_dataset: Change the embeddings dataset.
        get_sampling_rate: Get the sampling rate of the model.
        print_model_details: Print details of the model.
        quick_synthesize: Customize pipeline method for quick synthesis.
        change_dataset_split: Change dataset split (train, validation, test).
        load_custom_embedding: Load a custom speaker embedding (xvector) for the text.

    Usage:
        >>> speechT5 = SpeechT5Wrapper()
        >>> result = speechT5("Hello, how are you?")
        >>> speechT5.save_speech(result)
        >>> print("Speech saved successfully!")



    """

    def __init__(
        self,
        model_name="microsoft/speecht5_tts",
        vocoder_name="microsoft/speecht5_hifigan",
        dataset_name="Matthijs/cmu-arctic-xvectors",
    ):
        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self.dataset_name = dataset_name
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_name
        )
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_name)
        self.embeddings_dataset = load_dataset(
            self.dataset_name, split="validation"
        )

    def __call__(self, text: str, speaker_id: float = 7306):
        """Call the model on some text and return the speech."""
        speaker_embedding = torch.tensor(
            self.embeddings_dataset[speaker_id]["xvector"]
        ).unsqueeze(0)
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(
            inputs["input_ids"],
            speaker_embedding,
            vocoder=self.vocoder,
        )
        return speech

    def save_speech(self, speech, filename="speech.wav"):
        """Save Speech to a file."""
        sf.write(filename, speech.numpy(), samplerate=16000)

    def set_model(self, model_name: str):
        """Set the model to a new model."""
        self.model_name = model_name
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_name
        )

    def set_vocoder(self, vocoder_name):
        """Set the vocoder to a new vocoder."""
        self.vocoder_name = vocoder_name
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_name)

    def set_embeddings_dataset(self, dataset_name):
        """Set the embeddings dataset to a new dataset."""
        self.dataset_name = dataset_name
        self.embeddings_dataset = load_dataset(
            self.dataset_name, split="validation"
        )

    # Feature 1: Get sampling rate
    def get_sampling_rate(self):
        """Get sampling rate of the model."""
        return 16000

    # Feature 2: Print details of the model
    def print_model_details(self):
        """Print details of the model."""
        print(f"Model Name: {self.model_name}")
        print(f"Vocoder Name: {self.vocoder_name}")

    # Feature 3: Customize pipeline method for quick synthesis
    def quick_synthesize(self, text):
        """Customize pipeline method for quick synthesis."""
        synthesiser = pipeline("text-to-speech", self.model_name)
        speech = synthesiser(text)
        return speech

    # Feature 4: Change dataset split (train, validation, test)
    def change_dataset_split(self, split="train"):
        """Change dataset split (train, validation, test)."""
        self.embeddings_dataset = load_dataset(
            self.dataset_name, split=split
        )

    # Feature 5: Load a custom speaker embedding (xvector) for the text
    def load_custom_embedding(self, xvector):
        """Load a custom speaker embedding (xvector) for the text."""
        return torch.tensor(xvector).unsqueeze(0)


# if __name__ == "__main__":
#     speechT5 = SpeechT5Wrapper()
#     result = speechT5("Hello, how are you?")
#     speechT5.save_speech(result)
#     print("Speech saved successfully!")
