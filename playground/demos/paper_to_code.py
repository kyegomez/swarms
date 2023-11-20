from swarms.structs import Flow, SequentialWorkflow
from swarms.models import OpenAIChat, Anthropic

# llm
llm = OpenAIChat()
llm2 = Anthropic()

# 2 Flows, one that creates an algorithmic pseuedocode and another that creates the pytorch code
flow1 = Flow(llm2, max_loops=1)
flow2 = Flow(llm, max_loops=1)

# SequentialWorkflow
workflow = SequentialWorkflow(
    [flow1, flow2],
    max_loops=1,
    name="Paper to Code",
    autosave=True,
    description="This workflow takes a paper and converts it to code.",
)
