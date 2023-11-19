from swarms.models.biogpt import BioGPTWrapper

model = BioGPTWrapper()

out = model("The patient has a fever")

print(out)
