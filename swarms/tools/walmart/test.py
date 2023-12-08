from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "walmart", "http://127.0.0.1:8079/tools/walmart/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)

qa = STQuestionAnswerer()
agent = qa.load_tools(tools_name, tools_config)

agent(
    "I want to have Kung Pao Chicken tonight. Please help me buy some onions on walmart."
)
