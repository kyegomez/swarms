from swarms.structs.workflow import Workflow, StringTask
from swarms.models import OpenAIChat


llm = OpenAIChat()


workflow = Workflow(llm)
