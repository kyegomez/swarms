from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "douban", "http://127.0.0.1:8079/tools/douban-film/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
# tools_name, tools_config = load_single_tools()
print(tools_name, tools_config)

qa = STQuestionAnswerer()

agent = qa.load_tools(tools_name, tools_config)

agent("有哪些即将上映的中国喜剧电影？哪些是大家最想看的前5部？")
agent("想去电影院看一些国产电影，有评分高的吗？输出3部")
agent("帮我介绍下《深海》这部电影")
