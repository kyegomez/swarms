from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = (
    "baidu-translation",
    "http://127.0.0.1:8079/tools/baidu-translation/",
)
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()
print(tools_config)
agent = qa.load_tools(tools_name, tools_config)

answer = agent("Translate this sentence into Chinese(zh): Hello, world!")
print(answer)
