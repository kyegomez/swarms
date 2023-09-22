from swarms import Worker

node = Worker(
    openai_api_key="",
    ai_name="Optimus Prime",

)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)


# from swarms import Workflow
# from swarms.tools.autogpt import ChatOpenAI

# workflow = Workflow(ChatOpenAI)

# workflow.add("What's the weather in miami")
# workflow.add("Provide details for {{ parent_output }}")
# workflow.add("Summarize the above information: {{ parent_output}}")

# workflow.run()
