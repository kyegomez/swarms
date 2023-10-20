# LLMs
from swarms.models.anthropic import Anthropic
from swarms.models.petals import Petals
from swarms.models.mistral import Mistral
from swarms.models.openai_models import OpenAI, AzureOpenAI, OpenAIChat
from swarms.models.zephyr import Zephyr
from swarms.models.biogpt import BioGPT

# MultiModal Models
from swarms.models.idefics import Idefics
from swarms.models.kosmos_two import Kosmos
from swarms.models.vilt import Vilt
from swarms.models.nougat import Nougat
from swarms.models.layoutlm_document_qa import LayoutLMDocumentQA

# from swarms.models.fuyu import Fuyu # Not working, wait until they update


__all__ = [
    "Anthropic",
    "Petals",
    "Mistral",
    "OpenAI",
    "AzureOpenAI",
    "OpenAIChat",
    "Zephyr",
    "Idefics",
    "Kosmos",
    "Vilt",
    "Nougat",
    "LayoutLMDocumentQA",
]
