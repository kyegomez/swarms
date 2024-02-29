r"""
BioGPT
Pre-trained language models have attracted increasing attention in the biomedical domain,
inspired by their great success in the general natural language domain.
Among the two main branches of pre-trained language models in the general language domain,
i.e. BERT (and its variants) and GPT (and its variants),
the first one has been extensively studied in the biomedical domain, such as BioBERT and PubMedBERT.
While they have achieved great success on a variety of discriminative downstream biomedical tasks,
the lack of generation ability constrains their application scope.
In this paper, we propose BioGPT, a domain-specific generative Transformer language model
pre-trained on large-scale biomedical literature.
We evaluate BioGPT on six biomedical natural language processing tasks
and demonstrate that our model outperforms previous models on most tasks.
Especially, we get 44.98%, 38.42% and 40.76% F1 score on BC5CDR, KD-DTI and DDI
end-to-end relation extraction tasks, respectively, and 78.2% accuracy on PubMedQA,
creating a new record. Our case study on text generation further demonstrates the
advantage of BioGPT on biomedical literature to generate fluent descriptions for biomedical terms.


@article{10.1093/bib/bbac409,
    author = {Luo, Renqian and Sun, Liai and Xia, Yingce and Qin, Tao and Zhang, Sheng and Poon, Hoifung and Liu, Tie-Yan},
    title = "{BioGPT: generative pre-trained transformer for biomedical text generation and mining}",
    journal = {Briefings in Bioinformatics},
    volume = {23},
    number = {6},
    year = {2022},
    month = {09},
    issn = {1477-4054},
    doi = {10.1093/bib/bbac409},
    url = {https://doi.org/10.1093/bib/bbac409},
    note = {bbac409},
    eprint = {https://academic.oup.com/bib/article-pdf/23/6/bbac409/47144271/bbac409.pdf},
}
"""

import torch
from transformers import (
    BioGptForCausalLM,
    BioGptTokenizer,
    pipeline,
    set_seed,
)


class BioGPT:
    """
    A wrapper class for the BioGptForCausalLM model from the transformers library.

    Attributes:
        model_name (str): Name of the pretrained model.
        model (BioGptForCausalLM): The pretrained BioGptForCausalLM model.
        tokenizer (BioGptTokenizer): The tokenizer for the BioGptForCausalLM model.

    Methods:
        __call__: Generate text based on the given input.
        get_features: Get the features of a given text.
        beam_search_decoding: Generate text using beam search decoding.
        set_pretrained_model: Set a new tokenizer and model.
        get_config: Get the model's configuration.
        save_model: Save the model and tokenizer to a directory.
        load_from_path: Load a model and tokenizer from a directory.
        print_model: Print the model's architecture.

    Usage:
        >>> from swarms.models.biogpt import BioGPTWrapper
        >>> model = BioGPTWrapper()
        >>> out = model("The patient has a fever")
        >>> print(out)


    """

    def __init__(
        self,
        model_name: str = "microsoft/biogpt",
        max_length: int = 500,
        num_return_sequences: int = 5,
        do_sample: bool = True,
        min_length: int = 100,
    ):
        """
        Initialize the wrapper class with a model name.

        Args:
            model_name (str): Name of the pretrained model. Default is "microsoft/biogpt".
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample
        self.min_length = min_length

        self.model = BioGptForCausalLM.from_pretrained(
            self.model_name
        )
        self.tokenizer = BioGptTokenizer.from_pretrained(
            self.model_name
        )

    def __call__(self, text: str):
        """
        Generate text based on the given input.

        Args:
            text (str): The input text to generate from.
            max_length (int): Maximum length of the generated text.
            num_return_sequences (int): Number of sequences to return.
            do_sample (bool): Whether or not to use sampling in generation.

        Returns:
            list[dict]: A list of generated texts.
        """
        set_seed(42)
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        out = generator(
            text,
            max_length=self.max_length,
            num_return_sequences=self.num_return_sequences,
            do_sample=self.do_sample,
        )

        return out[0]["generated_text"]

    def get_features(self, text):
        """
        Get the features of a given text.

        Args:
            text (str): Input text.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Model output.
        """
        encoded_input = self.tokenizer(text, return_tensors="pt")
        return self.model(**encoded_input)

    def beam_search_decoding(
        self,
        sentence,
        num_beams=5,
        early_stopping=True,
    ):
        """
        Generate text using beam search decoding.

        Args:
            sentence (str): The input sentence to generate from.
            min_length (int): Minimum length of the generated text.
            max_length (int): Maximum length of the generated text.
            num_beams (int): Number of beams for beam search.
            early_stopping (bool): Whether to stop early during beam search.

        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        set_seed(42)
        with torch.no_grad():
            beam_output = self.model.generate(
                **inputs,
                min_length=self.min_length,
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
            )
        return self.tokenizer.decode(
            beam_output[0], skip_special_tokens=True
        )

    # Feature 1: Set a new tokenizer and model
    def set_pretrained_model(self, model_name):
        """
        Set a new tokenizer and model.

        Args:
            model_name (str): Name of the pretrained model.
        """
        self.model_name = model_name
        self.model = BioGptForCausalLM.from_pretrained(
            self.model_name
        )
        self.tokenizer = BioGptTokenizer.from_pretrained(
            self.model_name
        )

    # Feature 2: Get the model's config details
    def get_config(self):
        """
        Get the model's configuration.

        Returns:
            PretrainedConfig: The configuration of the model.
        """
        return self.model.config

    # Feature 3: Save the model and tokenizer to disk
    def save_model(self, path):
        """
        Save the model and tokenizer to a directory.

        Args:
            path (str): Path to the directory.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # Feature 4: Load a model from a custom path
    def load_from_path(self, path):
        """
        Load a model and tokenizer from a directory.

        Args:
            path (str): Path to the directory.
        """
        self.model = BioGptForCausalLM.from_pretrained(path)
        self.tokenizer = BioGptTokenizer.from_pretrained(path)

    # Feature 5: Print the model's architecture
    def print_model(self):
        """
        Print the model's architecture.
        """
        print(self.model)
