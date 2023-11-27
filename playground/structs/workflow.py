from swarms.structs.workflow import Workflow
from swarms.models import OpenAIChat


llm = OpenAIChat()


workflow = Workflow(llm)
