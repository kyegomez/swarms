from swarms.models import OpenAIChat
from swarms.structs import Workflow


llm = OpenAIChat(
    openai_api_key="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"
)

workflow = Workflow(llm)

workflow.add("What's the weather in miami")

workflow.run()