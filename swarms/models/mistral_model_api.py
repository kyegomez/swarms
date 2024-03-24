from swarms.models.popular_llms import OpenAIChat


class MistralAPILLM(OpenAIChat):
    def __init__(self, url):
        super().__init__()
        self.openai_proxy_url = url

    def __call__(self, task: str):
        super().__call__(task)
