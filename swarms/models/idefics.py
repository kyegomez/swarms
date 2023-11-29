import torch
from transformers import AutoProcessor, IdeficsForVisionText2Text


class Idefics:
    """

    A class for multimodal inference using pre-trained models from the Hugging Face Hub.

    Attributes
    ----------
    device : str
        The device to use for inference.
    checkpoint : str, optional
        The name of the pre-trained model checkpoint (default is "HuggingFaceM4/idefics-9b-instruct").
    processor : transformers.PreTrainedProcessor
        The pre-trained processor.
    max_length : int
        The maximum length of the generated text.
    chat_history : list
        The chat history.

    Methods
    -------
    infer(prompts, batched_mode=True)
        Generates text based on the provided prompts.
    chat(user_input)
        Engages in a continuous bidirectional conversation based on the user input.
    set_checkpoint(checkpoint)
        Changes the model checkpoint.
    set_device(device)
        Changes the device used for inference.
    set_max_length(max_length)
        Changes the maximum length of the generated text.
    clear_chat_history()
        Clears the chat history.


    # Usage
    ```
    from swarms.models import idefics

    model = idefics()

    user_input = "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
    response = model.chat(user_input)
    print(response)

    user_input = "User: And who is that? https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
    response = model.chat(user_input)
    print(response)

    model.set_checkpoint("new_checkpoint")
    model.set_device("cpu")
    model.set_max_length(200)
    model.clear_chat_history()
    ```

    """

    def __init__(
        self,
        checkpoint="HuggingFaceM4/idefics-9b-instruct",
        device=None,
        torch_dtype=torch.bfloat16,
        max_length=100,
    ):
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(checkpoint)

        self.max_length = max_length

        self.chat_history = []

    def run(self, prompts, batched_mode=True):
        """
        Generates text based on the provided prompts.

        Parameters
        ----------
            prompts : list
                A list of prompts. Each prompt is a list of text strings and images.
            batched_mode : bool, optional
                Whether to process the prompts in batched mode. If True, all prompts are
                processed together. If False, only the first prompt is processed (default is True).

        Returns
        -------
            list
                A list of generated text strings.
        """
        inputs = (
            self.processor(
                prompts,
                add_end_of_utterance_token=False,
                return_tensors="pt",
            ).to(self.device)
            if batched_mode
            else self.processor(prompts[0], return_tensors="pt").to(
                self.device
            )
        )

        exit_condition = self.processor.tokenizer(
            "<end_of_utterance>", add_special_tokens=False
        ).input_ids

        bad_words_ids = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image"],
            add_special_tokens=False,
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_length=self.max_length,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_text

    def __call__(self, prompts, batched_mode=True):
        """
        Generates text based on the provided prompts.

        Parameters
        ----------
            prompts : list
                A list of prompts. Each prompt is a list of text strings and images.
            batched_mode : bool, optional
                Whether to process the prompts in batched mode.
                If True, all prompts are processed together.
                If False, only the first prompt is processed (default is True).

        Returns
        -------
            list
                A list of generated text strings.
        """
        inputs = (
            self.processor(
                prompts,
                add_end_of_utterance_token=False,
                return_tensors="pt",
            ).to(self.device)
            if batched_mode
            else self.processor(prompts[0], return_tensors="pt").to(
                self.device
            )
        )

        exit_condition = self.processor.tokenizer(
            "<end_of_utterance>", add_special_tokens=False
        ).input_ids

        bad_words_ids = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image"],
            add_special_tokens=False,
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_length=self.max_length,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_text

    def chat(self, user_input):
        """
        Engages in a continuous bidirectional conversation based on the user input.

        Parameters
        ----------
            user_input : str
                The user input.

        Returns
        -------
            str
                The model's response.
        """
        self.chat_history.append(user_input)

        prompts = [self.chat_history]

        response = self.run(prompts)[0]

        self.chat_history.append(response)

        return response

    def set_checkpoint(self, checkpoint):
        """
        Changes the model checkpoint.

        Parameters
        ----------
            checkpoint : str
                The name of the new pre-trained model checkpoint.
        """
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def set_device(self, device):
        """
        Changes the device used for inference.

        Parameters
        ----------
            device : str
                The new device to use for inference.
        """
        self.device = device
        self.model.to(self.device)

    def set_max_length(self, max_length):
        """Set max_length"""
        self.max_length = max_length

    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []
