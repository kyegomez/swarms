from swarms.agents.base import agent
from swarms.structs.nonlinear_worfklow import NonLinearWorkflow, Task

prompt = "develop a feedforward network in pytorch"
prompt2 = "Develop a self attention using pytorch"

task1 = Task("task1", prompt)
task2 = Task("task2", prompt2, parents=[task1])

# add tasks to workflow
workflow = NonLinearWorkflow(agent)

# add tasks to tree
workflow.add(task1)
workflow.add(task2)

# run
workflow.run()
