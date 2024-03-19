from langchain_community.chat_models.azure_openai import (
    AzureChatOpenAI,
)
from langchain_community.chat_models.openai import (
    ChatOpenAI as OpenAIChat,
)
from langchain_community.llms import (
    Anthropic,
    Cohere,
    MosaicML,
    OpenAI,
    Replicate,
)


class AnthropicChat(Anthropic):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class CohereChat(Cohere):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class MosaicMLChat(MosaicML):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class OpenAILLM(OpenAI):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class ReplicateLLM(Replicate):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class AzureOpenAILLM(AzureChatOpenAI):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class OpenAIChatLLM(OpenAIChat):
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
