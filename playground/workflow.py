from swarms import Workflow
from swarms.models import ChatOpenAI

workflow = Workflow(ChatOpenAI)

workflow.add("What's the weather in miami")
workflow.add("Provide details for {{ parent_output }}")
workflow.add("Summarize the above information: {{ parent_output}}")

workflow.run()
