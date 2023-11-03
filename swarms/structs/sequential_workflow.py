"""
Sequential Workflow

from swarms.models import OpenAIChat, Mistral
from swarms.structs import SequentialWorkflow


llm = OpenAIChat(openai_api_key="")
mistral = Mistral()

# Max loops will run over the sequential pipeline twice
workflow = SequentialWorkflow(max_loops=2)

workflow.add("What's the weather in miami", llm)

workflow.add("Create a report on these metrics", mistral)

workflow.run()

"""
