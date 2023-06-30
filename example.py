from swarms import boss_node

#create a task
task = boss_node.create_task(objective="Write a research paper on the impact of climate change on global agriculture")

#execute the teask
boss_node.execute_task(task)

