from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = (
    "nllb-translation",
    "http://127.0.0.1:8079/tools/nllb-translation/",
)
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()
print(tools_config)
agent = qa.load_tools(tools_name, tools_config)

answer = agent("Translate this sentence into Chinese(zho_Hans): Hello, world!")
print(answer)
