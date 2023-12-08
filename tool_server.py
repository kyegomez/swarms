from swarms.tools.serve import ToolServer


def run_tool_server():
    # def load_weather_tool():
    # WEATHER_API_KEYS = os.environ.get('WEATHER_API_KEYS', None)
    # if not WEATHER_API_KEYS:
    # raise RuntimeError("WEATHER_API_KEYS not provided, please register one from https://www.weatherapi.com/ and add it to environment variables.")
    # server.load_tool("weather", {"subscription_key": WEATHER_API_KEYS})

    # def load_database_tool():
    #     server.load_tool("database")

    # def load_db_diag_tool():
    #     server.load_tool("db_diag")

    def load_chemical_prop_tool():
        server.load_tool("chemical-prop")

    def load_douban_tool():
        server.load_tool("douban-film")

    def load_wikipedia_tool():
        server.load_tool("wikipedia")

    # def load_wikidata_tool():
    #     server.load_tool("wikidata")

    # def load_travel_tool():
    #     server.load_tool("travel")

    # def load_wolframalpha_tool():
    #     WOLFRAMALPH_APP_ID = os.environ.get("WOLFRAMALPH_APP_ID", None)
    #     if not WOLFRAMALPH_APP_ID:
    #         raise RuntimeError("WOLFRAMALPH_APP_ID not provided, please register one from https://products.wolframalpha.com/api/ and add it to environment variables.")
    #     server.load_tool("wolframalpha", {"subscription_key": WOLFRAMALPH_APP_ID})

    # def load_bing_search_tool():
    #     BING_SUBSCRIPT_KEY = os.environ.get('BING_SUBSCRIPT_KEY', None)
    #     if not BING_SUBSCRIPT_KEY:
    #         raise RuntimeError("Bing search key not provided, please register one from https://www.microsoft.com/en-us/bing/apis/bing-web-search-api and add it to environment variables.")
    #     server.load_tool("bing_search", {"subscription_key": BING_SUBSCRIPT_KEY})

    # def load_office_ppt_tool():
    #     server.load_tool("office-ppt")

    # def load_alpha_vantage_tool():
    #     ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', None)
    #     if not ALPHA_VANTAGE_KEY:
    #         raise RuntimeError("Stock key not provided, please register one from https://www.alphavantage.co/support/#api-key and add it to environment variables.")
    #     server.load_tool("stock", {"subscription_key": ALPHA_VANTAGE_KEY})

    # def load_map_tool():
    #     BING_MAP_KEY = os.environ.get('BING_MAP_KEY', None)
    #     if not BING_MAP_KEY:
    #         raise RuntimeError("Bing map key not provided, please register one from https://www.bingmapsportal.com/ and add it to environment variables.")
    #     server.load_tool("bing_map", {"subscription_key": BING_MAP_KEY})

    # baidu map tool
    # BAIDU_SECRET_KEY = os.environ.get('BAIDU_SECRET_KEY', None)
    # BAIDU_MAP_KEY = os.environ.get('BAIDU_MAP_KEY', None)
    # if not BAIDU_SECRET_KEY or not BAIDU_MAP_KEY:
    #     raise RuntimeError("Baidu map key not provided, please register one from https://lbsyun.baidu.com/apiconsole/key and add it to environment variables.")
    # server.load_tool("baidu_map", {"subscription_key": BAIDU_MAP_KEY, "baidu_secret_key": BAIDU_SECRET_KEY})

    # def load_rapidapi_tool():
    #     RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', None)
    #     if not RAPIDAPI_KEY:
    #     if not RAPIDAPI_KEY:
    #         raise RuntimeError("RAPIDAPI_KEY not provided, please register one from https://rapidapi.com/ and add it to environment variables.")
    #     server.load_tool("zillow", {"subscription_key": RAPIDAPI_KEY})
    #     server.load_tool("airbnb", {"subscription_key": RAPIDAPI_KEY})
    #     server.load_tool("job_search", {"subscription_key": RAPIDAPI_KEY})

    # def load_nllb_translation_tool():
    #     server.load_tool("nllb-translation")

    # def load_baidu_translation_tool():
    #     server.load_tool("baidu-translation")

    def load_tutorial_tool():
        server.load_tool("tutorial")

    def load_file_operation_tool():
        server.load_tool("file_operation")

    def load_meta_analysis_tool():
        server.load_tool("meta_analysis")

    def load_code_interpreter_tool():
        server.load_tool("code_interpreter")

    def load_arxiv_tool():
        server.load_tool("arxiv")

    # def load_google_places_tool():
    #     GPLACES_API_KEY = os.environ.get('GPLACES_API_KEY', '')
    #     if not GPLACES_API_KEY:
    #         raise RuntimeError("GPLACES_API_KEY not provided, please register one from https://developers.google.com/maps/documentation/elevation/get-api-key and add it to environment variables.")
    #     server.load_tool("google_places", {"subscription_key": GPLACES_API_KEY})

    # def load_google_serper_tool():
    #     SERPER_API_KEY = os.environ.get('SERPER_API_KEY', None)
    #     if not SERPER_API_KEY:
    #         raise RuntimeError("SERPER_API_KEY not provided, please register one from https://serper.dev and add it to environment variables.")
    #     server.load_tool("google_serper", {"subscription_key": SERPER_API_KEY})
    #     server.load_tool("google_scholar", {"subscription_key": SERPER_API_KEY})
    #     server.load_tool("walmart", {"subscription_key": SERPER_API_KEY})

    def load_python_tool():
        server.load_tool("python")

    # def load_sceneXplain_tool():
    #     SCENEX_API_KEY = os.environ.get('SCENEX_API_KEY', None)
    #     if not SCENEX_API_KEY:
    #         raise RuntimeError("SCENEX_API_KEY is not provided. Please sign up for a free account at https://scenex.jina.ai/, create a new API key, and add it to environment variables.")
    #     server.load_tool("sceneXplain", {"subscription_key": SCENEX_API_KEY})

    def load_shell_tool():
        server.load_tool("shell")

    # def load_image_generation_tool():
    #     STEAMSHIP_API_KEY = os.environ.get('STEAMSHIP_API_KEY', None)
    #     if not STEAMSHIP_API_KEY:
    #         raise RuntimeError("STEAMSHIP_API_KEY is not provided. Please sign up for a free account at https://steamship.com/account/api, create a new API key, and add it to environment variables.")
    #     server.load_tool("image_generation")

    # def load_hugging_tools():
    #     HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', None)
    #     if not HUGGINGFACE_API_KEY:
    #         raise RuntimeError("Huggingface api key (access tokens) not provided, please register one from https://huggingface.co/ and add it to environment variables.")
    #     server.load_tool("hugging_tools")

    def load_gradio_tools():
        server.load_tool("gradio_tools")

    server = ToolServer()
    print(server.list_tools())

    # tool_choice = input("Enter 'ALL' to load all tools, or enter the specific tools you want to load (comma-separated): ")

    # load_weather_tool()
    # load_database_tool()
    # load_db_diag_tool()
    load_chemical_prop_tool()
    load_douban_tool()
    load_wikipedia_tool()
    # load_wikidata_tool()
    # load_wolframalpha_tool()
    # load_bing_search_tool()
    # load_office_ppt_tool()
    # load_alpha_vantage_tool()
    # load_map_tool()
    # load_rapidapi_tool()
    # load_nllb_translation_tool()
    # load_baidu_translation_tool()
    load_tutorial_tool()
    load_file_operation_tool()
    load_meta_analysis_tool()
    load_code_interpreter_tool()
    load_arxiv_tool()
    # load_google_places_tool()
    # load_google_serper_tool()
    load_python_tool()
    # load_sceneXplain_tool()
    load_shell_tool()
    # load_image_generation_tool()
    # load_hugging_tools()
    load_gradio_tools()
    # load_travel_tool()

    server.serve()


if __name__ == "__main__":
    run_tool_server()
