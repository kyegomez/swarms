from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = (
    "image_generation",
    "http://127.0.0.1:8079/tools/image_generation/",
)
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("How would you visualize a girl dancing ballet?")
