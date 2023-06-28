from swarms.agents.workers.auto_agent import agent

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tree of Thoughts",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    human_in_the_loop=True, # Set to True if you want to add feedback at each step.
)

agent.chain.verbose = True



tree_of_thoughts_prompt = """

Imagine three different experts are answering this question. All experts will write down each chain of thought of each step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...


"""


#Input problem
input_problem = """


Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation
Possible next steps:


"""

agent.run([f"{tree_of_thoughts_prompt} {input_problem}"])