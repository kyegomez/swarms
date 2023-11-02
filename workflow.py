from swarms.models import OpenAIChat
from swarms.structs import Workflow


llm = OpenAIChat(openai_api_key="")

workflow = Workflow(llm)

workflow.add("What's the weather in miami")

workflow.run()
