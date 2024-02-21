import torch

from swarms.models.base_llm import AbstractLLM

if torch.cuda.is_available() or torch.cuda.device_count() > 0:
    # Download vllm with pip
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        print(f"[ERROR] [vLLM] {error}")
        raise error
else:
    from swarms.models.huggingface import HuggingfaceLLM as LLM

    SamplingParams = None


class vLLM(AbstractLLM):
    """vLLM model


    Args:
        model_name (str, optional): _description_. Defaults to "facebook/opt-13b".
        tensor_parallel_size (int, optional): _description_. Defaults to 4.
        trust_remote_code (bool, optional): _description_. Defaults to False.
        revision (str, optional): _description_. Defaults to None.
        temperature (float, optional): _description_. Defaults to 0.5.
        top_p (float, optional): _description_. Defaults to 0.95.
        *args: _description_.
        **kwargs: _description_.

    Methods:
        run: run the vLLM model

    Raises:
        error: _description_

    Examples:
    >>> from swarms.models.vllm import vLLM
    >>> vllm = vLLM()
    >>> vllm.run("Hello world!")


    """

    def __init__(
        self,
        model_name: str = "facebook/opt-13b",
        tensor_parallel_size: int = 4,
        trust_remote_code: bool = False,
        revision: str = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.top_p = top_p

        # LLM model
        self.llm = LLM(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            *args,
            **kwargs,
        )

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, *args, **kwargs
        )

    def run(self, task: str = None, *args, **kwargs):
        """Run the vLLM model

        Args:
            task (str, optional): _description_. Defaults to None.

        Raises:
            error: _description_

        Returns:
            _type_: _description_
        """
        try:
            return self.llm.generate(
                task, self.sampling_params, *args, **kwargs
            )
        except Exception as error:
            print(f"[ERROR] [vLLM] [run] {error}")
            raise error
