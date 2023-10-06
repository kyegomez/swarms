from swarms.models import OpenAIChat

llm = OpenAIChat(openai_api_key="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")
out = llm("Hello, I am a robot and I like to talk about robots.")