from swarms.tools.tools_controller import load_valid_tools, MTQuestionAnswerer
import jsonlines

# Choose the tools that you need
tools_mappings = {
    # "klarna": "https://www.klarna.com/",
    # "chemical-prop": "http://127.0.0.1:8079/tools/chemical-prop/",
    "wolframalpha": "http://127.0.0.1:8079/tools/wolframalpha/",
    # "meta_analysis": "http://127.0.0.1:8079/tools/meta_analysis/",
    # "map": "http://127.0.0.1:8079/tools/map/",
    # "douban": "http://127.0.0.1:8079/tools/douban-film/",
    # "weather": "http://127.0.0.1:8079/tools/weather/",
    "office-ppt": "http://127.0.0.1:8079/tools/office-ppt/",
    "wikipedia": "http://127.0.0.1:8079/tools/wikipedia/",
    # "nllb-translation": "http://127.0.0.1:8079/tools/nllb-translation/",
    "file_operation": "http://127.0.0.1:8079/tools/file_operation/",
    "bing_search": "http://127.0.0.1:8079/tools/bing_search/",
}

tools = load_valid_tools(tools_mappings)
qa = MTQuestionAnswerer(all_tools=tools)

agent = qa.build_runner()


agent(
    [
        "Who's the main actress of Titanic? What did she do apart from this film? Help me make slides with this information."
    ]
)
# agent(['I want to go to Berkeley for one-week vacation. Please help me recommend some tourisms, restaurants, as well as the recent weather conditions for the place.'])
# agent(["How many benzene rings are there in 9H-Carbazole-3-carboxaldehyde? and what is sin(x)*exp(x)'s plot, what is it integrated from 0 to 1? "])
