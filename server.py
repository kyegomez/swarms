import os
import warnings

from swarms.modelui.modules.block_requests import OpenMonkeyPatch, RequestBlocker
from swarms.modelui.modules.logging_colors import logger

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')

with RequestBlocker():
    import gradio as gr

import matplotlib

matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from threading import Lock

import yaml

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


import yaml
import gradio as gr
from swarms.tools.tools_controller import MTQuestionAnswerer, load_valid_tools
from swarms.tools.singletool import STQuestionAnswerer
from langchain.schema import AgentFinish
import os
import requests

from swarms.modelui.server import create_interface
from tool_server import run_tool_server
from threading import Thread
from multiprocessing import Process
import time

tool_server_flag = False
def start_tool_server():
    # server = Thread(target=run_tool_server)
    server = Process(target=run_tool_server)
    server.start()
    global tool_server_flag
    tool_server_flag = True


available_models = ["ChatGPT", "GPT-3.5"]
DEFAULTMODEL = "ChatGPT"  # "GPT-3.5"

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

valid_tools_info = []
all_tools_list = []

gr.close_all()

MAX_TURNS = 30
MAX_BOXES = MAX_TURNS * 2

return_msg = []
chat_history = ""

MAX_SLEEP_TIME = 40
def load_tools():
    global valid_tools_info
    global all_tools_list
    try:
        valid_tools_info = load_valid_tools(tools_mappings)
    except BaseException as e:
        print(repr(e))
    all_tools_list = sorted(list(valid_tools_info.keys()))
    return gr.update(choices=all_tools_list)

def set_environ(OPENAI_API_KEY: str,
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
                AMADEUS_KEY: str = "",):
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

def create_interface():

    title = 'Swarm Models'

    # Password authentication
    auth = []
    if shared.args.gradio_auth:
        auth.extend(x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip())
    if shared.args.gradio_auth_path:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            auth.extend(x.strip() for line in file for x in line.split(',') if x.strip())
    auth = [tuple(cred.split(':')) for cred in auth]

    # Import the extensions and execute their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # Force some events to be triggered on page load
    shared.persistent_interface_state.update({
        'loader': shared.args.loader or 'Transformers',
        'mode': shared.settings['mode'],
        'character_menu': shared.args.character or shared.settings['character'],
        'instruction_template': shared.settings['instruction_template'],
        'prompt_menu-default': shared.settings['prompt-default'],
        'prompt_menu-notebook': shared.settings['prompt-notebook'],
        'filter_by_loader': shared.args.loader or 'All'
    })

    if Path("cache/pfp_character.png").exists():
        Path("cache/pfp_character.png").unlink()

    # css/js strings
    css = ui.css
    js = ui.js
    css += apply_extensions('css')
    js += apply_extensions('js')

    # Interface state elements
    shared.input_elements = ui.list_interface_input_elements()



    # with gr.Blocks() as demo:
    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:
        with gr.Row():
            with gr.Column(scale=14):
                gr.Markdown("")
            with gr.Column(scale=1):
                gr.Image(show_label=False, show_download_button=False, value="images/swarmslogobanner.png")

        with gr.Tab("Models"):
            create_interface()

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

                    chatbot = gr.Chatbot(show_label=False, visible=True).style(height=600)
                    buttonClear = gr.Button("Clear History")
                    buttonStop = gr.Button("Stop", visible=False)

                with gr.Column(scale=1):
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

        # Interface state
        shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})

        # Audio notification
        if Path("notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="notification.mp3", elem_id="audio_notification", visible=False)

        # Floating menus for saving/deleting files
        ui_file_saving.create_ui()

        # Temporary clipboard for saving files
        shared.gradio['temporary_text'] = gr.Textbox(visible=False)

        # Text Generation tab
        ui_chat.create_ui()
        ui_default.create_ui()
        ui_notebook.create_ui()

        ui_parameters.create_ui(shared.settings['preset'])  # Parameters tab
        ui_model_menu.create_ui()  # Model tab
        training.create_ui()  # Training tab
        ui_session.create_ui()  # Session tab

        # Generation events
        ui_chat.create_event_handlers()
        ui_default.create_event_handlers()
        ui_notebook.create_event_handlers()

        # Other events
        ui_file_saving.create_event_handlers()
        ui_parameters.create_event_handlers()
        ui_model_menu.create_event_handlers()

        # Interface launch events
        if shared.settings['dark_theme']:
            shared.gradio['interface'].load(lambda: None, None, None, _js="() => document.getElementsByTagName('body')[0].classList.add('dark')")

        shared.gradio['interface'].load(lambda: None, None, None, _js=f"() => {{{js}}}")
        shared.gradio['interface'].load(None, gradio('show_controls'), None, _js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')
        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, gradio(ui.list_interface_input_elements()), show_progress=False)
        shared.gradio['interface'].load(chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

        extensions_module.create_extensions_tabs()  # Extensions tabs
        extensions_module.create_extensions_block()  # Extensions block

    # Launch the interface
    shared.gradio['interface'].queue(concurrency_count=64)
    with OpenMonkeyPatch():
        shared.gradio['interface'].launch(
            prevent_thread_lock=True,
            share=shared.args.share,
            server_name=None if not shared.args.listen else (shared.args.listen_host or '0.0.0.0'),
            server_port=shared.args.listen_port,
            inbrowser=shared.args.auto_launch,
            auth=auth or None,
            ssl_verify=False if (shared.args.ssl_keyfile or shared.args.ssl_certfile) else True,
            ssl_keyfile=shared.args.ssl_keyfile,
            ssl_certfile=shared.args.ssl_certfile
        )


if __name__ == "__main__":

    # Load custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path('settings.yaml').exists():
        settings_file = Path('settings.yaml')
    elif Path('settings.json').exists():
        settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        shared.settings.update(new_settings)

    # Fallback settings for models
    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Activate the extensions listed on settings.yaml
    extensions_module.available_extensions = utils.get_available_extensions()
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        p = Path(shared.model_name)
        if p.exists():
            model_name = p.parts[-1]
            shared.model_name = model_name
        else:
            model_name = shared.model_name

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    shared.generation_lock = Lock()

    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
