from langchain_community.chat_models.azure_openai import (
    AzureChatOpenAI,
)
from langchain_community.chat_models.openai import (
    ChatOpenAI as OpenAIChat,
)
from langchain.llms.anthropic import Anthropic
from langchain.llms.cohere import Cohere
from langchain.llms.mosaicml import MosaicML
from langchain.llms.openai import OpenAI  # , OpenAIChat, AzureOpenAI
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.llms.replicate import Replicate
from langchain_community.llms.fireworks import Fireworks  # noqa: F401


class Anthropic(Anthropic):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class CohereChat(Cohere):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class MosaicMLChat(MosaicML):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class OpenAILLM(OpenAI):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class ReplicateChat(Replicate):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class AzureOpenAILLM(AzureChatOpenAI):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class OpenAIChatLLM(OpenAIChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        out = self.invoke(*args, **kwargs)
        return out.content.strip()

    def run(self, *args, **kwargs):
        out = self.invoke(*args, **kwargs)
        return out.content.strip()


class OctoAIChat(OctoAIEndpoint):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
