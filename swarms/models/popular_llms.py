from langchain_community.chat_models.azure_openai import (
    AzureChatOpenAI,
)
from langchain_community.chat_models.openai import (
    ChatOpenAI as OpenAIChat,
)
from langchain_community.llms.anthropic import Anthropic
from langchain_community.llms.cohere import Cohere
from langchain_community.llms.mosaicml import MosaicML
from langchain_community.llms.openai import (
    OpenAI,
)  # , OpenAIChat, AzureOpenAI
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_community.llms.replicate import Replicate
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


class FireWorksAI(Fireworks):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
