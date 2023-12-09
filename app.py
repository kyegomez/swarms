import os
import time
from functools import partial
from pathlib import Path
from threading import Lock
import warnings
import json

from swarms.modelui.modules.block_requests import OpenMonkeyPatch, RequestBlocker
from swarms.modelui.modules.logging_colors import logger
from swarms.modelui.server import create_interface

from vllm import LLM 

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')

with RequestBlocker():
    import gradio as gr

import matplotlib

matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import swarms.modelui.modules.extensions as extensions_module
from swarms.modelui.modules import (
    chat,
    shared,
    training,
    ui,
    ui_chat,
    ui_default,
    ui_file_saving,
    ui_model_menu,
    ui_notebook,
    ui_parameters,
    ui_session,
    utils
)
from swarms.modelui.modules.extensions import apply_extensions
from swarms.modelui.modules.LoRA import add_lora_to_model
from swarms.modelui.modules.models import load_model
from swarms.modelui.modules.models_settings import (
    get_fallback_settings,
    get_model_metadata,
    update_model_parameters
)
from swarms.modelui.modules.utils import gradio

import gradio as gr
from swarms.tools.tools_controller import MTQuestionAnswerer, load_valid_tools
from swarms.tools.singletool import STQuestionAnswerer
from langchain.schema import AgentFinish
import requests

from swarms.modelui.server import create_interface
from tool_server import run_tool_server
from threading import Thread
from multiprocessing import Process
import time
from langchain.llms import VLLM

tool_server_flag = False
def start_tool_server():
    # server = Thread(target=run_tool_server)
    server = Process(target=run_tool_server)
    server.start()
    global tool_server_flag
    tool_server_flag = True


DEFAULTMODEL = "ChatGPT"  # "GPT-3.5"

# Read the model/ directory and get the list of models
model_dir = Path("./models/")
available_models = ["ChatGPT", "GPT-3.5"] + [f.name for f in model_dir.iterdir() if f.is_dir()]

tools_mappings = {
    "klarna": "https://www.klarna.com/",
    "weather": "http://127.0.0.1:8079/tools/weather/",
    # "database": "http://127.0.0.1:8079/tools/database/",
    # "db_diag": "http://127.0.0.1:8079/tools/db_diag/",
    "chemical-prop": "http://127.0.0.1:8079/tools/chemical-prop/",
    "douban-film": "http://127.0.0.1:8079/tools/douban-film/",
    "wikipedia": "http://127.0.0.1:8079/tools/wikipedia/",
    # "wikidata": "http://127.0.0.1:8079/tools/kg/wikidata/",
    "wolframalpha": "http://127.0.0.1:8079/tools/wolframalpha/",
    "bing_search": "http://127.0.0.1:8079/tools/bing_search/",
    "office-ppt": "http://127.0.0.1:8079/tools/office-ppt/",
    "stock": "http://127.0.0.1:8079/tools/stock/",
    "bing_map": "http://127.0.0.1:8079/tools/map.bing_map/",
    # "baidu_map": "http://127.0.0.1:8079/tools/map/baidu_map/",
    "zillow": "http://127.0.0.1:8079/tools/zillow/",
    "airbnb": "http://127.0.0.1:8079/tools/airbnb/",
    "job_search": "http://127.0.0.1:8079/tools/job_search/",
    # "baidu-translation": "http://127.0.0.1:8079/tools/translation/baidu-translation/",
    # "nllb-translation": "http://127.0.0.1:8079/tools/translation/nllb-translation/",
    "tutorial": "http://127.0.0.1:8079/tools/tutorial/",
    "file_operation": "http://127.0.0.1:8079/tools/file_operation/",
    "meta_analysis": "http://127.0.0.1:8079/tools/meta_analysis/",
    "code_interpreter": "http://127.0.0.1:8079/tools/code_interpreter/",
    "arxiv": "http://127.0.0.1:8079/tools/arxiv/",
    "google_places": "http://127.0.0.1:8079/tools/google_places/",
    "google_serper": "http://127.0.0.1:8079/tools/google_serper/",
    "google_scholar": "http://127.0.0.1:8079/tools/google_scholar/",
    "python": "http://127.0.0.1:8079/tools/python/",
    "sceneXplain": "http://127.0.0.1:8079/tools/sceneXplain/",
    "shell": "http://127.0.0.1:8079/tools/shell/",
    "image_generation": "http://127.0.0.1:8079/tools/image_generation/",
    "hugging_tools": "http://127.0.0.1:8079/tools/hugging_tools/",
    "gradio_tools": "http://127.0.0.1:8079/tools/gradio_tools/",
    "travel": "http://127.0.0.1:8079/tools/travel",
    "walmart": "http://127.0.0.1:8079/tools/walmart",
}

# data = json.load(open('swarms/tools/openai.json')) # Load the JSON file
# items = data['items'] # Get the list of items

# for plugin in items: # Iterate over items, not data
#     url = plugin['manifest']['api']['url']
#     tool_name = plugin['namespace']
#     tools_mappings[tool_name] = url[:-len('/.well-known/openai.yaml')]

# print(tools_mappings)

valid_tools_info = []
all_tools_list = []

gr.close_all()

MAX_TURNS = 30
MAX_BOXES = MAX_TURNS * 2

return_msg = []
chat_history = ""

MAX_SLEEP_TIME = 40

def download_model(model_url: str, memory_utilization: int , model_dir: str):
    model_name = model_url.split('/')[-1]
    # Download the model using VLLM
    vllm_model = VLLM(
        model=model_url,
        trust_remote_code=True,
        gpu_memory_utilization=memory_utilization,
        download_dir=model_dir
    )
    # Add the downloaded model to the available_models list
    available_models.append((model_name, vllm_model))
    # Update the dropdown choices with the new available_models list
    model_chosen.update(choices=available_models)

valid_tools_info = {}

import gradio as gr
from swarms.tools.tools_controller import load_valid_tools, tools_mappings

def load_tools():
    global valid_tools_info
    global all_tools_list
    try:
        valid_tools_info = load_valid_tools(tools_mappings)
        print(f"valid_tools_info: {valid_tools_info}")  # Debugging line
    except BaseException as e:
        print(repr(e))
    all_tools_list = sorted(list(valid_tools_info.keys()))
    print(f"all_tools_list: {all_tools_list}")  # Debugging line
    return gr.update(choices=all_tools_list)


def set_environ(OPENAI_API_KEY: str = "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",
                WOLFRAMALPH_APP_ID: str = "",
                WEATHER_API_KEYS: str = "",
                BING_SUBSCRIPT_KEY: str = "",
                ALPHA_VANTAGE_KEY: str = "",
                BING_MAP_KEY: str = "",
                BAIDU_TRANSLATE_KEY: str = "",
                RAPIDAPI_KEY: str = "",
                SERPER_API_KEY: str = "",
                GPLACES_API_KEY: str = "",
                SCENEX_API_KEY: str = "",
                STEAMSHIP_API_KEY: str = "",
                HUGGINGFACE_API_KEY: str = "",
                AMADEUS_ID: str = "",
                AMADEUS_KEY: str = "",
            ):

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["WOLFRAMALPH_APP_ID"] = WOLFRAMALPH_APP_ID
    os.environ["WEATHER_API_KEYS"] = WEATHER_API_KEYS
    os.environ["BING_SUBSCRIPT_KEY"] = BING_SUBSCRIPT_KEY
    os.environ["ALPHA_VANTAGE_KEY"] = ALPHA_VANTAGE_KEY
    os.environ["BING_MAP_KEY"] = BING_MAP_KEY
    os.environ["BAIDU_TRANSLATE_KEY"] = BAIDU_TRANSLATE_KEY
    os.environ["RAPIDAPI_KEY"] = RAPIDAPI_KEY
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    os.environ["GPLACES_API_KEY"] = GPLACES_API_KEY
    os.environ["SCENEX_API_KEY"] = SCENEX_API_KEY
    os.environ["STEAMSHIP_API_KEY"] = STEAMSHIP_API_KEY
    os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
    os.environ["AMADEUS_ID"] = AMADEUS_ID
    os.environ["AMADEUS_KEY"] = AMADEUS_KEY
    if not tool_server_flag:
        start_tool_server()
        time.sleep(MAX_SLEEP_TIME)
    return gr.update(value="OK!")

def show_avatar_imgs(tools_chosen):
    if len(tools_chosen) == 0:
        tools_chosen = list(valid_tools_info.keys())
    img_template = '<a href="{}" style="float: left"> <img style="margin:5px" src="{}.png" width="24" height="24" alt="avatar" /> {} </a>'
    imgs = [valid_tools_info[tool]['avatar'] for tool in tools_chosen if valid_tools_info[tool]['avatar'] != None]
    imgs = ' '.join([img_template.format(img, img, tool) for img, tool in zip(imgs, tools_chosen)])
    return [gr.update(value='<span class="">' + imgs + '</span>', visible=True), gr.update(visible=True)]

def answer_by_tools(question, tools_chosen, model_chosen):
    global return_msg
    return_msg += [(question, None), (None, '...')]
    yield [gr.update(visible=True, value=return_msg), gr.update(), gr.update()]
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

    if len(tools_chosen) == 0:  # if there is no tools chosen, we use all todo (TODO: What if the pool is too large.)
        tools_chosen = list(valid_tools_info.keys())

    if len(tools_chosen) == 1:
        answerer = STQuestionAnswerer(OPENAI_API_KEY.strip(), stream_output=True, llm=model_chosen)
        agent_executor = answerer.load_tools(tools_chosen[0], valid_tools_info[tools_chosen[0]],
                                             prompt_type="react-with-tool-description", return_intermediate_steps=True)
    else:
        answerer = MTQuestionAnswerer(OPENAI_API_KEY.strip(),
                                      load_valid_tools({k: tools_mappings[k] for k in tools_chosen}),
                                      stream_output=True, llm=model_chosen)

        agent_executor = answerer.build_runner()

    global chat_history
    chat_history += "Question: " + question + "\n"
    question = chat_history
    for inter in agent_executor(question):
        if isinstance(inter, AgentFinish): continue
        result_str = []
        return_msg.pop()
        if isinstance(inter, dict):
            result_str.append("<font color=red>Answer:</font> {}".format(inter['output']))
            chat_history += "Answer:" + inter['output'] + "\n"
            result_str.append("...")
        else:
            try:
                not_observation = inter[0].log
            except:
                print(inter[0])
                not_observation = inter[0]
            if not not_observation.startswith('Thought:'):
                not_observation = "Thought: " + not_observation
            chat_history += not_observation
            not_observation = not_observation.replace('Thought:', '<font color=green>Thought: </font>')
            not_observation = not_observation.replace('Action:', '<font color=purple>Action: </font>')
            not_observation = not_observation.replace('Action Input:', '<font color=purple>Action Input: </font>')
            result_str.append("{}".format(not_observation))
            result_str.append("<font color=blue>Action output:</font>\n{}".format(inter[1]))
            chat_history += "\nAction output:" + inter[1] + "\n"
            result_str.append("...")
        return_msg += [(None, result) for result in result_str]
        yield [gr.update(visible=True, value=return_msg), gr.update(), gr.update()]
    return_msg.pop()
    if return_msg[-1][1].startswith("<font color=red>Answer:</font> "):
        return_msg[-1] = (return_msg[-1][0], return_msg[-1][1].replace("<font color=red>Answer:</font> ",
                                                                       "<font color=green>Final Answer:</font> "))
    yield [gr.update(visible=True, value=return_msg), gr.update(visible=True), gr.update(visible=False)]


def retrieve(tools_search):
    if tools_search == "":
        return gr.update(choices=all_tools_list)
    else:
        url = "http://127.0.0.1:8079/retrieve"
        param = {
            "query": tools_search
        }
        response = requests.post(url, json=param)
        result = response.json()
        retrieved_tools = result["tools"]
        return gr.update(choices=retrieved_tools)

def clear_retrieve():
    return [gr.update(value=""), gr.update(choices=all_tools_list)]


def clear_history():
    global return_msg
    global chat_history
    return_msg = []
    chat_history = ""
    yield gr.update(visible=True, value=return_msg)

title = 'Swarm Models'

# css/js strings
css = ui.css
js = ui.js
css += apply_extensions('css')
js += apply_extensions('js')

# with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as demo:
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=14):
            gr.Markdown("")
        with gr.Column(scale=1):
            gr.Image(show_label=False, show_download_button=False, value="images/swarmslogobanner.png")

    with gr.Tab("Key setting"):
        OPENAI_API_KEY = gr.Textbox(label="OpenAI API KEY:", placeholder="sk-...", type="text")
        WOLFRAMALPH_APP_ID = gr.Textbox(label="Wolframalpha app id:", placeholder="Key to use wlframalpha", type="text")
        WEATHER_API_KEYS = gr.Textbox(label="Weather api key:", placeholder="Key to use weather api", type="text")
        BING_SUBSCRIPT_KEY = gr.Textbox(label="Bing subscript key:", placeholder="Key to use bing search", type="text")
        ALPHA_VANTAGE_KEY = gr.Textbox(label="Stock api key:", placeholder="Key to use stock api", type="text")
        BING_MAP_KEY = gr.Textbox(label="Bing map key:", placeholder="Key to use bing map", type="text")
        BAIDU_TRANSLATE_KEY = gr.Textbox(label="Baidu translation key:", placeholder="Key to use baidu translation", type="text")
        RAPIDAPI_KEY = gr.Textbox(label="Rapidapi key:", placeholder="Key to use zillow, airbnb and job search", type="text")
        SERPER_API_KEY = gr.Textbox(label="Serper key:", placeholder="Key to use google serper and google scholar", type="text")
        GPLACES_API_KEY = gr.Textbox(label="Google places key:", placeholder="Key to use google places", type="text")
        SCENEX_API_KEY = gr.Textbox(label="Scenex api key:", placeholder="Key to use sceneXplain", type="text")
        STEAMSHIP_API_KEY = gr.Textbox(label="Steamship api key:", placeholder="Key to use image generation", type="text")
        HUGGINGFACE_API_KEY = gr.Textbox(label="Huggingface api key:", placeholder="Key to use models in huggingface hub", type="text")
        AMADEUS_ID = gr.Textbox(label="Amadeus id:", placeholder="Id to use Amadeus", type="text")
        AMADEUS_KEY = gr.Textbox(label="Amadeus key:", placeholder="Key to use Amadeus", type="text")
        key_set_btn = gr.Button(value="Set keys!")


    with gr.Tab("Chat with Tool"):
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=0.85):
                        txt = gr.Textbox(show_label=False, placeholder="Question here. Use Shift+Enter to add new line.",
                                         lines=1).style(container=False)
                    with gr.Column(scale=0.15, min_width=0):
                        buttonChat = gr.Button("Chat")

                memory_utilization = gr.Slider(label="Memory Utilization:", min=0, max=1, step=0.1, default=0.5)
                
                chatbot = gr.Chatbot(show_label=False, visible=True).style(height=600)
                buttonClear = gr.Button("Clear History")
                buttonStop = gr.Button("Stop", visible=False)

            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_url = gr.Textbox(label="VLLM Model URL:", placeholder="URL to download VLLM model from Hugging Face", type="text");
                        buttonDownload = gr.Button("Download Model");
                        buttonDownload.click(fn=download_model, inputs=[model_url, memory_utilization]);
                        model_chosen = gr.Dropdown(
                            list(available_models), value=DEFAULTMODEL, multiselect=False, label="Model provided",
                            info="Choose the model to solve your question, Default means ChatGPT."
                )
                with gr.Row():
                    tools_search = gr.Textbox(
                        lines=1,
                        label="Tools Search",
                        placeholder="Please input some text to search tools.",
                    )
                    buttonSearch = gr.Button("Reset search condition")
                tools_chosen = gr.CheckboxGroup(
                    choices=all_tools_list,
                    value=["chemical-prop"],
                    label="Tools provided",
                    info="Choose the tools to solve your question.",
                )

        with gr.Tab("model"):
            create_inferance();
            def serve_iframe():
                return f'hi'

        # def serve_iframe():
        #     return "<iframe src='http://localhost:8000/shader.html' width='100%' height='400'></iframe>"

        # iface = gr.Interface(fn=serve_iframe, inputs=[], outputs=gr.outputs.HTML())

        key_set_btn.click(fn=set_environ, inputs=[
        OPENAI_API_KEY,
        WOLFRAMALPH_APP_ID,
        WEATHER_API_KEYS,
        BING_SUBSCRIPT_KEY,
        ALPHA_VANTAGE_KEY,
        BING_MAP_KEY,
        BAIDU_TRANSLATE_KEY,
        RAPIDAPI_KEY,
        SERPER_API_KEY,
        GPLACES_API_KEY,
        SCENEX_API_KEY,
        STEAMSHIP_API_KEY,
        HUGGINGFACE_API_KEY,
        AMADEUS_ID,
        AMADEUS_KEY,
    ], outputs=key_set_btn)
    key_set_btn.click(fn=load_tools, outputs=tools_chosen)

    tools_search.change(retrieve, tools_search, tools_chosen)
    buttonSearch.click(clear_retrieve, [], [tools_search, tools_chosen])

    txt.submit(lambda: [gr.update(value=''), gr.update(visible=False), gr.update(visible=True)], [],
               [txt, buttonClear, buttonStop])
    inference_event = txt.submit(answer_by_tools, [txt, tools_chosen, model_chosen], [chatbot, buttonClear, buttonStop])
    buttonChat.click(answer_by_tools, [txt, tools_chosen, model_chosen], [chatbot, buttonClear, buttonStop])
    buttonStop.click(lambda: [gr.update(visible=True), gr.update(visible=False)], [], [buttonClear, buttonStop],
                     cancels=[inference_event])
    buttonClear.click(clear_history, [], chatbot)

# demo.queue().launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=7001)
demo.queue().launch()



