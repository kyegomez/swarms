import logging

import torch
from numpy.linalg import norm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def cos_sim(a, b):
    return a @ b.T / (norm(a) * norm(b))


class JinaEmbeddings:
    """
    A class for running inference on a given model.

    Attributes:
        model_id (str): The ID of the model.
        device (str): The device to run the model on (either 'cuda' or 'cpu').
        max_length (int): The maximum length of the output sequence.
        quantize (bool, optional): Whether to use quantization. Defaults to False.
        quantization_config (dict, optional): The configuration for quantization.
        verbose (bool, optional): Whether to print verbose logs. Defaults to False.
        logger (logging.Logger, optional): The logger to use. Defaults to a basic logger.

    # Usage
    ```
    from swarms.models import JinaEmbeddings

    model = JinaEmbeddings()

    embeddings = model("Encode this text")

    print(embeddings)


    ```
    """

    def __init__(
        self,
        model_id: str,
        device: str = None,
        max_length: int = 500,
        quantize: bool = False,
        quantization_config: dict = None,
        verbose=False,
        # logger=None,
        distributed=False,
        decoding=False,
        cos_sim: bool = False,
        *args,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_id = model_id
        self.max_length = max_length
        self.verbose = verbose
        self.distributed = distributed
        self.decoding = decoding
        self.model, self.tokenizer = None, None
        # self.log = Logging()
        self.cos_sim = cos_sim

        if self.distributed:
            assert (
                torch.cuda.device_count() > 1
            ), "You need more than 1 gpu for distributed processing"

        bnb_config = None
        if quantize:
            if not quantization_config:
                quantization_config = {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                }
            bnb_config = BitsAndBytesConfig(**quantization_config)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

            self.model  # .to(self.device)
        except Exception as e:
            self.logger.error(
                f"Failed to load the model or the tokenizer: {e}"
            )
            raise

    def load_model(self):
        """Load the model"""
        if not self.model or not self.tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id
                )

                bnb_config = (
                    BitsAndBytesConfig(**self.quantization_config)
                    if self.quantization_config
                    else None
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                ).to(self.device)

                if self.distributed:
                    self.model = DDP(self.model)
            except Exception as error:
                self.logger.error(
                    "Failed to load the model or the tokenizer:"
                    f" {error}"
                )
                raise

    def run(self, task: str):
        """
        Generate a response based on the prompt text.

        Args:
        - task (str): Text to prompt the model.
        - max_length (int): Maximum length of the response.

        Returns:
        - Generated text (str).
        """
        self.load_model()

        max_length = self.max_length

        try:
            embeddings = self.model.encode(
                [task], max_length=max_length
            )

            if self.cos_sim:
                print(cos_sim(embeddings[0], embeddings[1]))
            else:
                return embeddings[0]
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise

    async def run_async(self, task: str, *args, **kwargs) -> str:
        """
        Run the model asynchronously

        Args:
            task (str): Task to run.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Examples:
        >>> mpt_instance = MPT('mosaicml/mpt-7b-storywriter', "EleutherAI/gpt-neox-20b", max_tokens=150)
        >>> mpt_instance("generate", "Once upon a time in a land far, far away...")
        'Once upon a time in a land far, far away...'
        >>> mpt_instance.batch_generate(["In the deep jungles,", "At the heart of the city,"], temperature=0.7)
        ['In the deep jungles,',
        'At the heart of the city,']
        >>> mpt_instance.freeze_model()
        >>> mpt_instance.unfreeze_model()

        """
        # Wrapping synchronous calls with async
        return self.run(task, *args, **kwargs)

    def __call__(self, task: str):
        """
        Generate a response based on the prompt text.

        Args:
        - task (str): Text to prompt the model.
        - max_length (int): Maximum length of the response.

        Returns:
        - Generated text (str).
        """
        self.load_model()

        max_length = self.max_length

        try:
            embeddings = self.model.encode(
                [task], max_length=max_length
            )

            if self.cos_sim:
                print(cos_sim(embeddings[0], embeddings[1]))
            else:
                return embeddings[0]
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise

    async def __call_async__(self, task: str, *args, **kwargs) -> str:
        """Call the model asynchronously""" ""
        return await self.run_async(task, *args, **kwargs)

    def save_model(self, path: str):
        """Save the model to a given path"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def gpu_available(self) -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available()

    def memory_consumption(self) -> dict:
        """Get the memory consumption of the GPU"""
        if self.gpu_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return {"allocated": allocated, "reserved": reserved}
        else:
            return {"error": "GPU not available"}
