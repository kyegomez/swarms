from typing import List, Union

from swarms.models.base_embedding_model import BaseEmbeddingModel
from swarms.models.base_llm import BaseLLM
from swarms.models.base_multimodal_model import BaseMultiModalModel
from swarms.models.fuyu import Fuyu  # noqa: E402
from swarms.models.gpt4_vision_api import GPT4VisionAPI  # noqa: E402
from swarms.models.huggingface import HuggingfaceLLM  # noqa: E402
from swarms.models.idefics import Idefics  # noqa: E402
from swarms.models.kosmos_two import Kosmos  # noqa: E402
from swarms.models.layoutlm_document_qa import LayoutLMDocumentQA
from swarms.models.llama3_hosted import llama3Hosted
from swarms.models.llava import LavaMultiModal  # noqa: E402
from swarms.models.nougat import Nougat  # noqa: E402
from swarms.models.openai_embeddings import OpenAIEmbeddings
from swarms.models.openai_function_caller import OpenAIFunctionCaller
from swarms.models.openai_tts import OpenAITTS  # noqa: E402
from swarms.models.palm import GooglePalm as Palm  # noqa: E402
from swarms.models.popular_llms import Anthropic as Anthropic
from swarms.models.popular_llms import (
    AzureOpenAILLM as AzureOpenAI,
)
from swarms.models.popular_llms import (
    CohereChat as Cohere,
)
from swarms.models.popular_llms import FireWorksAI, OctoAIChat
from swarms.models.popular_llms import (
    OpenAIChatLLM as OpenAIChat,
)
from swarms.models.popular_llms import (
    OpenAILLM as OpenAI,
)
from swarms.models.popular_llms import ReplicateChat as Replicate
from swarms.models.qwen import QwenVLMultiModal  # noqa: E402
from swarms.models.sampling_params import SamplingParams
from swarms.models.together import TogetherLLM  # noqa: E402
from swarms.models.vilt import Vilt  # noqa: E402
from swarms.structs.base_structure import BaseStructure
from swarms.utils.loguru_logger import logger

# New type BaseLLM and BaseEmbeddingModel and BaseMultimodalModel
omni_model_type = Union[
    BaseLLM, BaseEmbeddingModel, BaseMultiModalModel, callable
]
list_of_omni_model_type = List[omni_model_type]


models = [
    BaseLLM,
    BaseEmbeddingModel,
    BaseMultiModalModel,
    Fuyu,
    GPT4VisionAPI,
    HuggingfaceLLM,
    Idefics,
    Kosmos,
    LayoutLMDocumentQA,
    llama3Hosted,
    LavaMultiModal,
    Nougat,
    OpenAIEmbeddings,
    OpenAITTS,
    Palm,
    Anthropic,
    AzureOpenAI,
    Cohere,
    OctoAIChat,
    OpenAIChat,
    OpenAI,
    Replicate,
    QwenVLMultiModal,
    SamplingParams,
    TogetherLLM,
    Vilt,
    FireWorksAI,
    OpenAIFunctionCaller,
]


class ModelRouter(BaseStructure):
    """
    A router for managing multiple models.

    Attributes:
        model_router_id (str): The ID of the model router.
        model_router_description (str): The description of the model router.
        model_pool (List[omni_model_type]): The list of models in the model pool.

    Methods:
        check_for_models(): Checks if there are any models in the model pool.
        add_model(model: omni_model_type): Adds a model to the model pool.
        add_models(models: List[omni_model_type]): Adds multiple models to the model pool.
        get_model_by_name(model_name: str) -> omni_model_type: Retrieves a model from the model pool by its name.
        get_multiple_models_by_name(model_names: List[str]) -> List[omni_model_type]: Retrieves multiple models from the model pool by their names.
        get_model_pool() -> List[omni_model_type]: Retrieves the entire model pool.
        get_model_by_index(index: int) -> omni_model_type: Retrieves a model from the model pool by its index.
        get_model_by_id(model_id: str) -> omni_model_type: Retrieves a model from the model pool by its ID.
        dict() -> dict: Returns a dictionary representation of the model router.

    """

    def __init__(
        self,
        model_router_id: str = "model_router",
        model_router_description: str = "A router for managing multiple models.",
        model_pool: List[omni_model_type] = models,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_router_id = model_router_id
        self.model_router_description = model_router_description
        self.model_pool = model_pool
        self.verbose = verbose

        self.check_for_models()
        # self.refactor_model_class_if_invoke()

    def check_for_models(self):
        """
        Checks if there are any models in the model pool.

        Returns:
            None

        Raises:
            ValueError: If no models are found in the model pool.
        """
        if len(self.model_pool) == 0:
            raise ValueError("No models found in model pool.")

    def add_model(self, model: omni_model_type):
        """
        Adds a model to the model pool.

        Args:
            model (omni_model_type): The model to be added.

        Returns:
            str: A success message indicating that the model has been added to the model pool.
        """
        logger.info(f"Adding model {model.name} to model pool.")
        self.model_pool.append(model)
        return "Model successfully added to model pool."

    def add_models(self, models: List[omni_model_type]):
        """
        Adds multiple models to the model pool.

        Args:
            models (List[omni_model_type]): The models to be added.

        Returns:
            str: A success message indicating that the models have been added to the model pool.
        """
        logger.info("Adding models to model pool.")
        self.model_pool.extend(models)
        return "Models successfully added to model pool."

    # def query_model_from_langchain(self, model_name: str, *args, **kwargs):
    #     """
    #     Query a model from langchain community.

    #     Args:
    #         model_name (str): The name of the model.
    #         *args: Additional positional arguments to be passed to the model.
    #         **kwargs: Additional keyword arguments to be passed to the model.

    #     Returns:
    #         omni_model_type: The model object.

    #     Raises:
    #         ValueError: If the model with the given name is not found in the model pool.
    #     """
    #     from langchain_community.llms import __getattr__

    #     logger.info(
    #         f"Querying model {model_name} from langchain community."
    #     )
    #     model = __getattr__(model_name)(*args, **kwargs)
    #     model = self.refactor_model_class_if_invoke_class(model)

    #     return model

    def get_model_by_name(self, model_name: str) -> omni_model_type:
        """
        Retrieves a model from the model pool by its name.

        Args:
            model_name (str): The name of the model.

        Returns:
            omni_model_type: The model object.

        Raises:
            ValueError: If the model with the given name is not found in the model pool.
        """
        logger.info(f"Retrieving model {model_name} from model pool.")
        for model in self.model_pool:
            if model_name in [
                model.name,
                model.model_id,
                model.model_name,
            ]:
                return model
        raise ValueError(f"Model {model_name} not found in model pool.")

    def get_multiple_models_by_name(
        self, model_names: List[str]
    ) -> List[omni_model_type]:
        """
        Retrieves multiple models from the model pool by their names.

        Args:
            model_names (List[str]): The names of the models.

        Returns:
            List[omni_model_type]: The list of model objects.

        Raises:
            ValueError: If any of the models with the given names are not found in the model pool.
        """
        logger.info(
            f"Retrieving multiple models {model_names} from model pool."
        )
        models = []
        for model_name in model_names:
            models.append(self.get_model_by_name(model_name))
        return models

    def get_model_pool(self) -> List[omni_model_type]:
        """
        Retrieves the entire model pool.

        Returns:
            List[omni_model_type]: The list of model objects in the model pool.
        """
        return self.model_pool

    def get_model_by_index(self, index: int) -> omni_model_type:
        """
        Retrieves a model from the model pool by its index.

        Args:
            index (int): The index of the model in the model pool.

        Returns:
            omni_model_type: The model object.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.model_pool[index]

    def get_model_by_id(self, model_id: str) -> omni_model_type:
        """
        Retrieves a model from the model pool by its ID.

        Args:
            model_id (str): The ID of the model.

        Returns:
            omni_model_type: The model object.

        Raises:
            ValueError: If the model with the given ID is not found in the model pool.
        """
        name = model_id
        for model in self.model_pool:
            if (
                hasattr(model, "model_id")
                and name == model.model_id
                or hasattr(model, "model_name")
                and name == model.model_name
                or hasattr(model, "name")
                and name == model.name
                or hasattr(model, "model")
                and name == model.model
            ):
                return model
        raise ValueError(f"Model {model_id} not found in model pool.")

    def refactor_model_class_if_invoke(self):
        """
        Refactors the model class if it has an 'invoke' method.

        Checks to see if the model pool has a model with an 'invoke' method and refactors it to have a 'run' method and '__call__' method.

        Returns:
            str: A success message indicating that the model classes have been refactored.
        """
        for model in self.model_pool:
            if hasattr(model, "invoke"):
                model.run = model.invoke
                model.__call__ = model.invoke
                logger.info(
                    f"Refactored model {model.name} to have run and __call__ methods."
                )

                # Update the model in the model pool
                self.model_pool[self.model_pool.index(model)] = model

        return "Model classes successfully refactored."

    def refactor_model_class_if_invoke_class(
        self, model: callable, *args, **kwargs
    ) -> callable:
        """
        Refactors the model class if it has an 'invoke' method.

        Checks to see if the model pool has a model with an 'invoke' method and refactors it to have a 'run' method and '__call__' method.

        Returns:
            str: A success message indicating that the model classes have been refactored.
        """
        if hasattr(model, "invoke"):
            model.run = model.invoke
            model.__call__ = model.invoke
            logger.info(
                f"Refactored model {model.name} to have run and __call__ methods."
            )

        return model

    def find_model_by_name_and_run(
        self, model_name: str = None, task: str = None, *args, **kwargs
    ) -> str:
        """
        Finds a model by its name and runs a task on it.

        Args:
            model_name (str): The name of the model.
            task (str): The task to be run on the model.
            *args: Additional positional arguments to be passed to the task.
            **kwargs: Additional keyword arguments to be passed to the task.

        Returns:
            str: The result of running the task on the model.

        Raises:
            ValueError: If the model with the given name is not found in the model pool.
        """
        model = self.get_model_by_name(model_name)
        return model.run(task, *args, **kwargs)


# model = ModelRouter()
# print(model.to_dict())
# print(model.get_model_pool())
# print(model.get_model_by_index(0))
# print(model.get_model_by_id("stability-ai/stable-diffusion:"))
# # print(model.get_multiple_models_by_name(["gpt-4o", "gpt-4"]))
