from swarms import llama3Hosted

llama3 = llama3Hosted()

task = "What is the capital of France?"
response = llama3.run(task)
print(response)
