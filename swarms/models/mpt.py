import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class MPT7B:
    """
    MPT class for generating text using a pre-trained model.

    Args:
        model_name (str): Name of the model to use.
        tokenizer_name (str): Name of the tokenizer to use.
        max_tokens (int): Maximum number of tokens to generate.

    Attributes:
        model_name (str): Name of the model to use.
        tokenizer_name (str): Name of the tokenizer to use.
        tokenizer (transformers.AutoTokenizer): Tokenizer object.
        model (transformers.AutoModelForCausalLM): Model object.
        pipe (transformers.pipelines.TextGenerationPipeline): Text generation pipeline.
        max_tokens (int): Maximum number of tokens to generate.


    Examples:
    >>> mpt_instance = MPT('mosaicml/mpt-7b-storywriter', "EleutherAI/gpt-neox-20b", max_tokens=150)
    >>> mpt_instance("generate", "Once upon a time in a land far, far away...")
    'Once upon a time in a land far, far away...'


    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        max_tokens: int = 100,
    ):
        # Loading model and tokenizer details
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        config = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )

        # Initializing a text-generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cuda:0",
        )
        self.max_tokens = max_tokens

    def run(self, task: str, *args, **kwargs) -> str:
        """
        Run the model

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
        if task == "generate":
            return self.generate(*args, **kwargs)
        else:
            raise ValueError(f"Task '{task}' not recognized!")

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

    def generate(self, prompt: str) -> str:
        """

        Generate Text

        Args:
            prompt (str): Prompt to generate text from.

        Examples:


        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.pipe(
                prompt,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                use_cache=True,
            )[0]["generated_text"]

    async def generate_async(self, prompt: str) -> str:
        """Generate Async"""
        return self.generate(prompt)

    def __call__(self, task: str, *args, **kwargs) -> str:
        """Call the model"""
        return self.run(task, *args, **kwargs)

    async def __call_async__(self, task: str, *args, **kwargs) -> str:
        """Call the model asynchronously""" ""
        return await self.run_async(task, *args, **kwargs)

    def batch_generate(
        self, prompts: list, temperature: float = 1.0
    ) -> list:
        """Batch generate text"""
        self.logger.info(
            f"Generating text for {len(prompts)} prompts..."
        )
        results = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for prompt in prompts:
                result = self.pipe(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    use_cache=True,
                    temperature=temperature,
                )
                results.append(result[0]["generated_text"])
        return results

    def unfreeze_model(self):
        """Unfreeze the model"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.logger.info("Model has been unfrozen.")


# # Example usage:
# mpt_instance = MPT(
#     "mosaicml/mpt-7b-storywriter", "EleutherAI/gpt-neox-20b", max_tokens=150
# )

# # For synchronous calls
# print(mpt_instance("generate", "Once upon a time in a land far, far away..."))

# For asynchronous calls, use an event loop or similar async framework
# For example:
# # import asyncio
# # asyncio.run(mpt_instance.__call_async__("generate", "Once upon a time in a land far, far away..."))
# # Example usage:
# mpt_instance = MPT('mosaicml/mpt-7b-storywriter', "EleutherAI/gpt-neox-20b", max_tokens=150)

# # For synchronous calls
# print(mpt_instance("generate", "Once upon a time in a land far, far away..."))
# print(mpt_instance.batch_generate(["In the deep jungles,", "At the heart of the city,"], temperature=0.7))

# # Freezing and unfreezing the model
# mpt_instance.freeze_model()
# mpt_instance.unfreeze_model()
