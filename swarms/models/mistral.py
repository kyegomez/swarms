import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms.structs.message import Message
from swarms.models.base_llm import AbstractLLM

class Mistral(AbstractLLM):
    """
    Mistral is an all-new llm

    Args:
        ai_name (str, optional): Name of the AI. Defaults to "Mistral".
        system_prompt (str, optional): System prompt. Defaults to None.
        model_name (str, optional): Model name. Defaults to "mistralai/Mistral-7B-v0.1".
        device (str, optional): Device to use. Defaults to "cuda".
        use_flash_attention (bool, optional): Whether to use flash attention. Defaults to False.
        temperature (float, optional): Temperature. Defaults to 1.0.
        max_length (int, optional): Max length. Defaults to 100.
        do_sample (bool, optional): Whether to sample. Defaults to True.

    Usage:
    from swarms.models import Mistral

    model = Mistral(device="cuda", use_flash_attention=True, temperature=0.7, max_length=200)

    task = "My favourite condiment is"
    result = model.run(task)
    print(result)
    """

    def __init__(
        self,
        ai_name: str = "Node Model Agent",
        system_prompt: str = None,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str = "cuda",
        use_flash_attention: bool = False,
        temperature: float = 1.0,
        max_length: int = 100,
        do_sample: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()
        self.ai_name = ai_name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.device = device
        self.use_flash_attention = use_flash_attention
        self.temperature = temperature
        self.max_length = max_length
        self.do_sample = do_sample

        # Check if the specified device is available
        if not torch.cuda.is_available() and device == "cuda":
            raise ValueError(
                "CUDA is not available. Please choose a different"
                " device."
            )

        self.history = []

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, *args, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, *args, **kwargs
        )
        
        self.model.to(self.device)

    def run(self, task: str, *args, **kwargs):
        """Run the model on a given task."""

        try:
            model_inputs = self.tokenizer(
                [task], return_tensors="pt"
            ).to(self.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_length=self.max_length,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_length,
                **kwargs
            )
            output_text = self.tokenizer.batch_decode(generated_ids)[
                0
            ]
            return output_text
        except Exception as e:
            raise ValueError(f"Error running the model: {str(e)}")

    def chat(self, msg: str = None, streaming: bool = False):
        """
        Run chat

        Args:
            msg (str, optional): Message to send to the agent. Defaults to None.
            language (str, optional): Language to use. Defaults to None.
            streaming (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            str: Response from the agent

        Usage:
        --------------
        agent = MultiModalAgent()
        agent.chat("Hello")

        """

        # add users message to the history
        self.history.append(Message("User", msg))

        # process msg
        try:
            response = self.agent.run(msg)

            # add agent's response to the history
            self.history.append(Message("Agent", response))

            # if streaming is = True
            if streaming:
                return self._stream_response(response)
            else:
                response

        except Exception as error:
            error_message = f"Error processing message: {str(error)}"

            # add error to history
            self.history.append(Message("Agent", error_message))

            return error_message