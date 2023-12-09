from swarms.models.dalle3 import Dalle3

model = Dalle3()

task = "A painting of a dog"
img = model(task)
