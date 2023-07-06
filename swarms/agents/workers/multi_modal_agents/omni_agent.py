
import base64
import copy
from io import BytesIO
import io
import os
import random
import time
import traceback
import uuid
import requests
import re
import json
import logging
import argparse
import yaml
from PIL import Image, ImageDraw
from diffusers.utils import load_image
from pydub import AudioSegment
import threading
from queue import Queue
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS, cross_origin
from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model, count_tokens, get_max_context_length
from huggingface_hub.inference_api import InferenceApi
from huggingface_hub.inference_api import ALL_TASKS

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.default.yaml")
parser.add_argument("--mode", type=str, default="cli")
args = parser.parse_args()

if __name__ != "__main__":
    args.config = "configs/config.gradio.yaml"
    args.mode = "gradio"

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

os.makedirs("logs", exist_ok=True)
os.makedirs("public/images", exist_ok=True)
os.makedirs("public/audios", exist_ok=True)
os.makedirs("public/videos", exist_ok=True)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not config["debug"]:
    handler.setLevel(logging.CRITICAL)
logger.addHandler(handler)

log_file = config["log_file"]
if log_file:
    filehandler = logging.FileHandler(log_file)
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

LLM = config["model"]
use_completion = config["use_completion"]

# consistent: wrong msra model name 
LLM_encoding = LLM
if config["dev"] and LLM == "gpt-3.5-turbo":
    LLM_encoding = "text-davinci-003"
task_parsing_highlight_ids = get_token_ids_for_task_parsing(LLM_encoding)
choose_model_highlight_ids = get_token_ids_for_choose_model(LLM_encoding)

# ENDPOINT	MODEL NAME	
# /v1/chat/completions	gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301	
# /v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada

if use_completion:
    api_name = "completions"
else:
    api_name = "chat/completions"

API_TYPE = None
# priority: local > azure > openai
if "dev" in config and config["dev"]:
    API_TYPE = "local"
elif "azure" in config:
    API_TYPE = "azure"
elif "openai" in config:
    API_TYPE = "openai"
else:
    logger.warning(f"No endpoint specified in {args.config}. The endpoint will be set dynamically according to the client.")

if args.mode in ["test", "cli"]:
    assert API_TYPE, "Only server mode supports dynamic endpoint."

API_KEY = None
API_ENDPOINT = None
if API_TYPE == "local":
    API_ENDPOINT = f"{config['local']['endpoint']}/v1/{api_name}"
elif API_TYPE == "azure":
    API_ENDPOINT = f"{config['azure']['base_url']}/openai/deployments/{config['azure']['deployment_name']}/{api_name}?api-version={config['azure']['api_version']}"
    API_KEY = config["azure"]["api_key"]
elif API_TYPE == "openai":
    API_ENDPOINT = f"https://api.openai.com/v1/{api_name}"
    if config["openai"]["api_key"].startswith("sk-"):  # Check for valid OpenAI key in config file
        API_KEY = config["openai"]["api_key"]
    elif "OPENAI_API_KEY" in os.environ and os.getenv("OPENAI_API_KEY").startswith("sk-"):  # Check for environment variable OPENAI_API_KEY
        API_KEY = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Incorrect OpenAI key. Please check your {args.config} file.")

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }

inference_mode = config["inference_mode"]

# check the local_inference_endpoint
Model_Server = None
if inference_mode!="huggingface":
    Model_Server = "http://" + config["local_inference_endpoint"]["host"] + ":" + str(config["local_inference_endpoint"]["port"])
    message = f"The server of local inference endpoints is not running, please start it first. (or using `inference_mode: huggingface` in {args.config} for a feature-limited experience)"
    try:
        r = requests.get(Model_Server + "/running")
        if r.status_code != 200:
            raise ValueError(message)
    except:
        raise ValueError(message)


parse_task_demos_or_presteps = open(config["demos_or_presteps"]["parse_task"], "r").read()
choose_model_demos_or_presteps = open(config["demos_or_presteps"]["choose_model"], "r").read()
response_results_demos_or_presteps = open(config["demos_or_presteps"]["response_results"], "r").read()

parse_task_prompt = config["prompt"]["parse_task"]
choose_model_prompt = config["prompt"]["choose_model"]
response_results_prompt = config["prompt"]["response_results"]

parse_task_tprompt = config["tprompt"]["parse_task"]
choose_model_tprompt = config["tprompt"]["choose_model"]
response_results_tprompt = config["tprompt"]["response_results"]

MODELS = [json.loads(line) for line in open("data/p0_models.jsonl", "r").readlines()]
MODELS_MAP = {}
for model in MODELS:
    tag = model["task"]
    if tag not in MODELS_MAP:
        MODELS_MAP[tag] = []
    MODELS_MAP[tag].append(model)
METADATAS = {}
for model in MODELS:
    METADATAS[model["id"]] = model

HUGGINGFACE_HEADERS = {}
if config["huggingface"]["token"] and config["huggingface"]["token"].startswith("hf_"):  # Check for valid huggingface token in config file
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {config['huggingface']['token']}",
    }
elif "HUGGINGFACE_ACCESS_TOKEN" in os.environ and os.getenv("HUGGINGFACE_ACCESS_TOKEN").startswith("hf_"):  # Check for environment variable HUGGINGFACE_ACCESS_TOKEN
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_ACCESS_TOKEN')}",
    }
else:
    raise ValueError(f"Incorrect HuggingFace token. Please check your {args.config} file.")

def convert_chat_to_completion(data):
    messages = data.pop('messages', [])
    tprompt = ""
    if messages[0]['role'] == "system":
        tprompt = messages[0]['content']
        messages = messages[1:]
    final_prompt = ""
    for message in messages:
        if message['role'] == "user":
            final_prompt += ("<im_start>"+ "user" + "\n" + message['content'] + "<im_end>\n")
        elif message['role'] == "assistant":
            final_prompt += ("<im_start>"+ "assistant" + "\n" + message['content'] + "<im_end>\n")
        else:
            final_prompt += ("<im_start>"+ "system" + "\n" + message['content'] + "<im_end>\n")
    final_prompt = tprompt + final_prompt
    final_prompt = final_prompt + "<im_start>assistant"
    data["prompt"] = final_prompt
    data['stop'] = data.get('stop', ["<im_end>"])
    data['max_tokens'] = data.get('max_tokens', max(get_max_context_length(LLM) - count_tokens(LLM_encoding, final_prompt), 1))
    return data

def send_request(data):
    api_key = data.pop("api_key")
    api_type = data.pop("api_type")
    api_endpoint = data.pop("api_endpoint")
    if use_completion:
        data = convert_chat_to_completion(data)
    if api_type == "openai":
        HEADER = {
            "Authorization": f"Bearer {api_key}"
        }
    elif api_type == "azure":
        HEADER = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
    else:
        HEADER = None
    response = requests.post(api_endpoint, json=data, headers=HEADER, proxies=PROXY)
    if "error" in response.json():
        return response.json()
    logger.debug(response.text.strip())
    if use_completion:
        return response.json()["choices"][0]["text"].strip()
    else:
        return response.json()["choices"][0]["message"]["content"].strip()

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'").replace('\n', ""))
    return text

def find_json(s):
    s = s.replace("\'", "\"")
    start = s.find("{")
    end = s.rfind("}")
    res = s[start:end+1]
    res = res.replace("\n", "")
    return res

def field_extract(s, field):
    try:
        field_rep = re.compile(f'{field}.*?:.*?"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    except:
        field_rep = re.compile(f'{field}:\ *"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    return extracted

def get_id_reason(choose_str):
    reason = field_extract(choose_str, "reason")
    id = field_extract(choose_str, "id")
    choose = {"id": id, "reason": reason}
    return id.strip(), reason.strip(), choose

def record_case(success, **args):
    if success:
        f = open("logs/log_success.jsonl", "a")
    else:
        f = open("logs/log_fail.jsonl", "a")
    log = args
    f.write(json.dumps(log) + "\n")
    f.close()

def image_to_bytes(img_url):
    img_byte = io.BytesIO()
    type = img_url.split(".")[-1]
    load_image(img_url).save(img_byte, format="png")
    img_data = img_byte.getvalue()
    return img_data

def resource_has_dep(command):
    args = command["args"]
    for _, v in args.items():
        if "<GENERATED>" in v:
            return True
    return False

def fix_dep(tasks):
    for task in tasks:
        args = task["args"]
        task["dep"] = []
        for k, v in args.items():
            if "<GENERATED>" in v:
                dep_task_id = int(v.split("-")[1])
                if dep_task_id not in task["dep"]:
                    task["dep"].append(dep_task_id)
        if len(task["dep"]) == 0:
            task["dep"] = [-1]
    return tasks

def unfold(tasks):
    flag_unfold_task = False
    try:
        for task in tasks:
            for key, value in task["args"].items():
                if "<GENERATED>" in value:
                    generated_items = value.split(",")
                    if len(generated_items) > 1:
                        flag_unfold_task = True
                        for item in generated_items:
                            new_task = copy.deepcopy(task)
                            dep_task_id = int(item.split("-")[1])
                            new_task["dep"] = [dep_task_id]
                            new_task["args"][key] = item
                            tasks.append(new_task)
                        tasks.remove(task)
    except Exception as e:
        print(e)
        traceback.print_exc()
        logger.debug("unfold task failed.")

    if flag_unfold_task:
        logger.debug(f"unfold tasks: {tasks}")
        
    return tasks

def chitchat(messages, api_key, api_type, api_endpoint):
    data = {
        "model": LLM,
        "messages": messages,
        "api_key": api_key,
        "api_type": api_type,
        "api_endpoint": api_endpoint
    }
    return send_request(data)

def parse_task(context, input, api_key, api_type, api_endpoint):
    demos_or_presteps = parse_task_demos_or_presteps
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": parse_task_tprompt})

    # cut chat logs
    start = 0
    while start <= len(context):
        history = context[start:]
        prompt = replace_slot(parse_task_prompt, {
            "input": input,
            "context": history 
        })
        messages.append({"role": "user", "content": prompt})
        history_text = "<im_end>\nuser<im_start>".join([m["content"] for m in messages])
        num = count_tokens(LLM_encoding, history_text)
        if get_max_context_length(LLM) - num > 800:
            break
        messages.pop()
        start += 2
    
    logger.debug(messages)
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0,
        "logit_bias": {item: config["logit_bias"]["parse_task"] for item in task_parsing_highlight_ids},
        "api_key": api_key,
        "api_type": api_type,
        "api_endpoint": api_endpoint
    }
    return send_request(data)

def choose_model(input, task, metas, api_key, api_type, api_endpoint):
    prompt = replace_slot(choose_model_prompt, {
        "input": input,
        "task": task,
        "metas": metas,
    })
    demos_or_presteps = replace_slot(choose_model_demos_or_presteps, {
        "input": input,
        "task": task,
        "metas": metas
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": choose_model_tprompt})
    messages.append({"role": "user", "content": prompt})
    logger.debug(messages)
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0,
        "logit_bias": {item: config["logit_bias"]["choose_model"] for item in choose_model_highlight_ids}, # 5
        "api_key": api_key,
        "api_type": api_type,
        "api_endpoint": api_endpoint
    }
    return send_request(data)


def response_results(input, results, api_key, api_type, api_endpoint):
    results = [v for k, v in sorted(results.items(), key=lambda item: item[0])]
    prompt = replace_slot(response_results_prompt, {
        "input": input,
    })
    demos_or_presteps = replace_slot(response_results_demos_or_presteps, {
        "input": input,
        "processes": results
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": response_results_tprompt})
    messages.append({"role": "user", "content": prompt})
    logger.debug(messages)
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0,
        "api_key": api_key,
        "api_type": api_type,
        "api_endpoint": api_endpoint
    }
    return send_request(data)

def huggingface_model_inference(model_id, data, task):
    task_url = f"https://api-inference.huggingface.co/models/{model_id}" # InferenceApi does not yet support some tasks
    inference = InferenceApi(repo_id=model_id, token=config["huggingface"]["token"])
    
    # NLP tasks
    if task == "question-answering":
        inputs = {"question": data["text"], "context": (data["context"] if "context" in data else "" )}
        result = inference(inputs)
    if task == "sentence-similarity":
        inputs = {"source_sentence": data["text1"], "target_sentence": data["text2"]}
        result = inference(inputs)
    if task in ["text-classification",  "token-classification", "text2text-generation", "summarization", "translation", "conversational", "text-generation"]:
        inputs = data["text"]
        result = inference(inputs)
    
    # CV tasks
    if task == "visual-question-answering" or task == "document-question-answering":
        img_url = data["image"]
        text = data["text"]
        img_data = image_to_bytes(img_url)
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        json_data = {}
        json_data["inputs"] = {}
        json_data["inputs"]["question"] = text
        json_data["inputs"]["image"] = img_base64
        result = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json=json_data).json()
        # result = inference(inputs) # not support

    if task == "image-to-image":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        # result = inference(data=img_data) # not support
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        r = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data)
        result = r.json()
        if "path" in result:
            result["generated image"] = result.pop("path")
    
    if task == "text-to-image":
        inputs = data["text"]
        img = inference(inputs)
        name = str(uuid.uuid4())[:4]
        img.save(f"public/images/{name}.png")
        result = {}
        result["generated image"] = f"/images/{name}.png"

    if task == "image-segmentation":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        image = Image.open(BytesIO(img_data))
        predicted = inference(data=img_data)
        colors = []
        for i in range(len(predicted)):
            colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 155))
        for i, pred in enumerate(predicted):
            label = pred["label"]
            mask = pred.pop("mask").encode("utf-8")
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode='r')
            mask = mask.convert('L')

            layer = Image.new('RGBA', mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        result = {}
        result["generated image"] = f"/images/{name}.jpg"
        result["predicted"] = predicted

    if task == "object-detection":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        predicted = inference(data=img_data)
        image = Image.open(BytesIO(img_data))
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=2)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        result = {}
        result["generated image"] = f"/images/{name}.jpg"
        result["predicted"] = predicted

    if task in ["image-classification"]:
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        result = inference(data=img_data)
 
    if task == "image-to-text":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        r = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data, proxies=PROXY)
        result = {}
        if "generated_text" in r.json()[0]:
            result["generated text"] = r.json()[0].pop("generated_text")
    
    # AUDIO tasks
    if task == "text-to-speech":
        inputs = data["text"]
        response = inference(inputs, raw_response=True)
        # response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json={"inputs": text})
        name = str(uuid.uuid4())[:4]
        with open(f"public/audios/{name}.flac", "wb") as f:
            f.write(response.content)
        result = {"generated audio": f"/audios/{name}.flac"}
    if task in ["automatic-speech-recognition", "audio-to-audio", "audio-classification"]:
        audio_url = data["audio"]
        audio_data = requests.get(audio_url, timeout=10).content
        response = inference(data=audio_data, raw_response=True)
        result = response.json()
        if task == "audio-to-audio":
            content = None
            type = None
            for k, v in result[0].items():
                if k == "blob":
                    content = base64.b64decode(v.encode("utf-8"))
                if k == "content-type":
                    type = "audio/flac".split("/")[-1]
            audio = AudioSegment.from_file(BytesIO(content))
            name = str(uuid.uuid4())[:4]
            audio.export(f"public/audios/{name}.{type}", format=type)
            result = {"generated audio": f"/audios/{name}.{type}"}
    return result

def local_model_inference(model_id, data, task):
    task_url = f"{Model_Server}/models/{model_id}"
    
    # contronlet
    if model_id.startswith("lllyasviel/sd-controlnet-"):
        img_url = data["image"]
        text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if model_id.endswith("-control"):
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
        
    if task == "text-to-video":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated video"] = results.pop("path")
        return results

    # NLP tasks
    if task == "question-answering" or task == "sentence-similarity":
        response = requests.post(task_url, json=data)
        return response.json()
    if task in ["text-classification",  "token-classification", "text2text-generation", "summarization", "translation", "conversational", "text-generation"]:
        response = requests.post(task_url, json=data)
        return response.json()

    # CV tasks
    if task == "depth-estimation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "image-segmentation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        results["generated image"] = results.pop("path")
        return results
    if task == "image-to-image":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "text-to-image":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "object-detection":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        predicted = response.json()
        if "error" in predicted:
            return predicted
        image = load_image(img_url)
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=2)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        results = {}
        results["generated image"] = f"/images/{name}.jpg"
        results["predicted"] = predicted
        return results
    if task in ["image-classification", "image-to-text", "document-question-answering", "visual-question-answering"]:
        img_url = data["image"]
        text = None
        if "text" in data:
            text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        return results
    # AUDIO tasks
    if task == "text-to-speech":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated audio"] = results.pop("path")
        return results
    if task in ["automatic-speech-recognition", "audio-to-audio", "audio-classification"]:
        audio_url = data["audio"]
        response = requests.post(task_url, json={"audio_url": audio_url})
        return response.json()


def model_inference(model_id, data, hosted_on, task):
    if hosted_on == "unknown":
        localStatusUrl = f"{Model_Server}/status/{model_id}"
        r = requests.get(localStatusUrl)
        logger.debug("Local Server Status: " + str(r.json()))
        if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
            hosted_on = "local"
        else:
            huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
            r = requests.get(huggingfaceStatusUrl, headers=HUGGINGFACE_HEADERS, proxies=PROXY)
            logger.debug("Huggingface Status: " + str(r.json()))
            if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
                hosted_on = "huggingface"
    try:
        if hosted_on == "local":
            inference_result = local_model_inference(model_id, data, task)
        elif hosted_on == "huggingface":
            inference_result = huggingface_model_inference(model_id, data, task)
    except Exception as e:
        print(e)
        traceback.print_exc()
        inference_result = {"error":{"message": str(e)}}
    return inference_result


def get_model_status(model_id, url, headers, queue = None):
    endpoint_type = "huggingface" if "huggingface" in url else "local"
    if "huggingface" in url:
        r = requests.get(url, headers=headers, proxies=PROXY)
    else:
        r = requests.get(url)
    if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
        if queue:
            queue.put((model_id, True, endpoint_type))
        return True
    else:
        if queue:
            queue.put((model_id, False, None))
        return False

def get_avaliable_models(candidates, topk=5):
    all_available_models = {"local": [], "huggingface": []}
    threads = []
    result_queue = Queue()

    for candidate in candidates:
        model_id = candidate["id"]

        if inference_mode != "local":
            huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
            thread = threading.Thread(target=get_model_status, args=(model_id, huggingfaceStatusUrl, HUGGINGFACE_HEADERS, result_queue))
            threads.append(thread)
            thread.start()
        
        if inference_mode != "huggingface" and config["local_deployment"] != "minimal":
            localStatusUrl = f"{Model_Server}/status/{model_id}"
            thread = threading.Thread(target=get_model_status, args=(model_id, localStatusUrl, {}, result_queue))
            threads.append(thread)
            thread.start()
        
    result_count = len(threads)
    while result_count:
        model_id, status, endpoint_type = result_queue.get()
        if status and model_id not in all_available_models:
            all_available_models[endpoint_type].append(model_id)
        if len(all_available_models["local"] + all_available_models["huggingface"]) >= topk:
            break
        result_count -= 1

    for thread in threads:
        thread.join()

    return all_available_models

def collect_result(command, choose, inference_result):
    result = {"task": command}
    result["inference result"] = inference_result
    result["choose model result"] = choose
    logger.debug(f"inference result: {inference_result}")
    return result


def run_task(input, command, results, api_key, api_type, api_endpoint):
    id = command["id"]
    args = command["args"]
    task = command["task"]
    deps = command["dep"]
    if deps[0] != -1:
        dep_tasks = [results[dep] for dep in deps]
    else:
        dep_tasks = []
    
    logger.debug(f"Run task: {id} - {task}")
    logger.debug("Deps: " + json.dumps(dep_tasks))

    if deps[0] != -1:
        if "image" in args and "<GENERATED>-" in args["image"]:
            resource_id = int(args["image"].split("-")[1])
            if "generated image" in results[resource_id]["inference result"]:
                args["image"] = results[resource_id]["inference result"]["generated image"]
        if "audio" in args and "<GENERATED>-" in args["audio"]:
            resource_id = int(args["audio"].split("-")[1])
            if "generated audio" in results[resource_id]["inference result"]:
                args["audio"] = results[resource_id]["inference result"]["generated audio"]
        if "text" in args and "<GENERATED>-" in args["text"]:
            resource_id = int(args["text"].split("-")[1])
            if "generated text" in results[resource_id]["inference result"]:
                args["text"] = results[resource_id]["inference result"]["generated text"]

    text = image = audio = None
    for dep_task in dep_tasks:
        if "generated text" in dep_task["inference result"]:
            text = dep_task["inference result"]["generated text"]
            logger.debug("Detect the generated text of dependency task (from results):" + text)
        elif "text" in dep_task["task"]["args"]:
            text = dep_task["task"]["args"]["text"]
            logger.debug("Detect the text of dependency task (from args): " + text)
        if "generated image" in dep_task["inference result"]:
            image = dep_task["inference result"]["generated image"]
            logger.debug("Detect the generated image of dependency task (from results): " + image)
        elif "image" in dep_task["task"]["args"]:
            image = dep_task["task"]["args"]["image"]
            logger.debug("Detect the image of dependency task (from args): " + image)
        if "generated audio" in dep_task["inference result"]:
            audio = dep_task["inference result"]["generated audio"]
            logger.debug("Detect the generated audio of dependency task (from results): " + audio)
        elif "audio" in dep_task["task"]["args"]:
            audio = dep_task["task"]["args"]["audio"]
            logger.debug("Detect the audio of dependency task (from args): " + audio)

    if "image" in args and "<GENERATED>" in args["image"]:
        if image:
            args["image"] = image
    if "audio" in args and "<GENERATED>" in args["audio"]:
        if audio:
            args["audio"] = audio
    if "text" in args and "<GENERATED>" in args["text"]:
        if text:
            args["text"] = text

    for resource in ["image", "audio"]:
        if resource in args and not args[resource].startswith("public/") and len(args[resource]) > 0 and not args[resource].startswith("http"):
            args[resource] = f"public/{args[resource]}"
    
    if "-text-to-image" in command['task'] and "text" not in args:
        logger.debug("control-text-to-image task, but text is empty, so we use control-generation instead.")
        control = task.split("-")[0]
        
        if control == "seg":
            task = "image-segmentation"
            command['task'] = task
        elif control == "depth":
            task = "depth-estimation"
            command['task'] = task
        else:
            task = f"{control}-control"

    command["args"] = args
    logger.debug(f"parsed task: {command}")

    if task.endswith("-text-to-image") or task.endswith("-control"):
        if inference_mode != "huggingface":
            if task.endswith("-text-to-image"):
                control = task.split("-")[0]
                best_model_id = f"lllyasviel/sd-controlnet-{control}"
            else:
                best_model_id = task
            hosted_on = "local"
            reason = "ControlNet is the best model for this task."
            choose = {"id": best_model_id, "reason": reason}
            logger.debug(f"chosen model: {choose}")
        else:
            logger.warning(f"Task {command['task']} is not available. ControlNet need to be deployed locally.")
            record_case(success=False, **{"input": input, "task": command, "reason": f"Task {command['task']} is not available. ControlNet need to be deployed locally.", "op":"message"})
            inference_result = {"error": f"service related to ControlNet is not available."}
            results[id] = collect_result(command, "", inference_result)
            return False
    elif task in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]: # ChatGPT Can do
        best_model_id = "ChatGPT"
        reason = "ChatGPT performs well on some NLP tasks as well."
        choose = {"id": best_model_id, "reason": reason}
        messages = [{
            "role": "user",
            "content": f"[ {input} ] contains a task in JSON format {command}. Now you are a {command['task']} system, the arguments are {command['args']}. Just help me do {command['task']} and give me the result. The result must be in text form without any urls."
        }]
        response = chitchat(messages, api_key, api_type, api_endpoint)
        results[id] = collect_result(command, choose, {"response": response})
        return True
    else:
        if task not in MODELS_MAP:
            logger.warning(f"no available models on {task} task.")
            record_case(success=False, **{"input": input, "task": command, "reason": f"task not support: {command['task']}", "op":"message"})
            inference_result = {"error": f"{command['task']} not found in available tasks."}
            results[id] = collect_result(command, "", inference_result)
            return False

        candidates = MODELS_MAP[task][:10]
        all_avaliable_models = get_avaliable_models(candidates, config["num_candidate_models"])
        all_avaliable_model_ids = all_avaliable_models["local"] + all_avaliable_models["huggingface"]
        logger.debug(f"avaliable models on {command['task']}: {all_avaliable_models}")

        if len(all_avaliable_model_ids) == 0:
            logger.warning(f"no available models on {command['task']}")
            record_case(success=False, **{"input": input, "task": command, "reason": f"no available models: {command['task']}", "op":"message"})
            inference_result = {"error": f"no available models on {command['task']} task."}
            results[id] = collect_result(command, "", inference_result)
            return False
            
        if len(all_avaliable_model_ids) == 1:
            best_model_id = all_avaliable_model_ids[0]
            hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
            reason = "Only one model available."
            choose = {"id": best_model_id, "reason": reason}
            logger.debug(f"chosen model: {choose}")
        else:
            cand_models_info = [
                {
                    "id": model["id"],
                    "inference endpoint": all_avaliable_models.get(
                        "local" if model["id"] in all_avaliable_models["local"] else "huggingface"
                    ),
                    "likes": model.get("likes"),
                    "description": model.get("description", "")[:config["max_description_length"]],
                    # "language": model.get("meta").get("language") if model.get("meta") else None,
                    "tags": model.get("meta").get("tags") if model.get("meta") else None,
                }
                for model in candidates
                if model["id"] in all_avaliable_model_ids
            ]

            choose_str = choose_model(input, command, cand_models_info, api_key, api_type, api_endpoint)
            logger.debug(f"chosen model: {choose_str}")
            try:
                choose = json.loads(choose_str)
                reason = choose["reason"]
                best_model_id = choose["id"]
                hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
            except Exception as e:
                logger.warning(f"the response [ {choose_str} ] is not a valid JSON, try to find the model id and reason in the response.")
                choose_str = find_json(choose_str)
                best_model_id, reason, choose  = get_id_reason(choose_str)
                hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
    inference_result = model_inference(best_model_id, args, hosted_on, command['task'])

    if "error" in inference_result:
        logger.warning(f"Inference error: {inference_result['error']}")
        record_case(success=False, **{"input": input, "task": command, "reason": f"inference error: {inference_result['error']}", "op":"message"})
        results[id] = collect_result(command, choose, inference_result)
        return False
    
    results[id] = collect_result(command, choose, inference_result)
    return True

def chat_huggingface(messages, api_key, api_type, api_endpoint, return_planning = False, return_results = False):
    start = time.time()
    context = messages[:-1]
    input = messages[-1]["content"]
    logger.info("*"*80)
    logger.info(f"input: {input}")

    task_str = parse_task(context, input, api_key, api_type, api_endpoint)

    if "error" in task_str:
        record_case(success=False, **{"input": input, "task": task_str, "reason": f"task parsing error: {task_str['error']['message']}", "op":"report message"})
        return {"message": task_str["error"]["message"]}

    task_str = task_str.strip()
    logger.info(task_str)

    try:
        tasks = json.loads(task_str)
    except Exception as e:
        logger.debug(e)
        response = chitchat(messages, api_key, api_type, api_endpoint)
        record_case(success=False, **{"input": input, "task": task_str, "reason": "task parsing fail", "op":"chitchat"})
        return {"message": response}
    
    if task_str == "[]":  # using LLM response for empty task
        record_case(success=False, **{"input": input, "task": [], "reason": "task parsing fail: empty", "op": "chitchat"})
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    if len(tasks) == 1 and tasks[0]["task"] in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]:
        record_case(success=True, **{"input": input, "task": tasks, "reason": "chitchat tasks", "op": "chitchat"})
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    tasks = unfold(tasks)
    tasks = fix_dep(tasks)
    logger.debug(tasks)
    
    if return_planning:
        return tasks

    results = {}
    threads = []
    tasks = tasks[:]
    d = dict()
    retry = 0
    while True:
        num_thread = len(threads)
        for task in tasks:
            # logger.debug(f"d.keys(): {d.keys()}, dep: {dep}")
            for dep_id in task["dep"]:
                if dep_id >= task["id"]:
                    task["dep"] = [-1]
                    break
            dep = task["dep"]
            if dep[0] == -1 or len(list(set(dep).intersection(d.keys()))) == len(dep):
                tasks.remove(task)
                thread = threading.Thread(target=run_task, args=(input, task, d, api_key, api_type, api_endpoint))
                thread.start()
                threads.append(thread)
        if num_thread == len(threads):
            time.sleep(0.5)
            retry += 1
        if retry > 160:
            logger.debug("User has waited too long, Loop break.")
            break
        if len(tasks) == 0:
            break
    for thread in threads:
        thread.join()
    
    results = d.copy()

    logger.debug(results)
    if return_results:
        return results
    
    response = response_results(input, results, api_key, api_type, api_endpoint).strip()

    end = time.time()
    during = end - start

    answer = {"message": response}
    record_case(success=True, **{"input": input, "task": task_str, "results": results, "response": response, "during": during, "op":"response"})
    logger.info(f"response: {response}")
    return answer

def test():
    # single round examples
    inputs = [
        "Given a collection of image A: /examples/a.jpg, B: /examples/b.jpg, C: /examples/c.jpg, please tell me how many zebras in these picture?"
        "Can you give me a picture of a small bird flying in the sky with trees and clouds. Generate a high definition image if possible.",
        "Please answer all the named entities in the sentence: Iron Man is a superhero appearing in American comic books published by Marvel Comics. The character was co-created by writer and editor Stan Lee, developed by scripter Larry Lieber, and designed by artists Don Heck and Jack Kirby.",
        "please dub for me: 'Iron Man is a superhero appearing in American comic books published by Marvel Comics. The character was co-created by writer and editor Stan Lee, developed by scripter Larry Lieber, and designed by artists Don Heck and Jack Kirby.'"
        "Given an image: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg, please answer the question: What is on top of the building?",
        "Please generate a canny image based on /examples/f.jpg"
        ]
        
    for input in inputs:
        messages = [{"role": "user", "content": input}]
        chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT, return_planning = False, return_results = False)
    
    # multi rounds example
    messages = [
        {"role": "user", "content": "Please generate a canny image based on /examples/f.jpg"},
        {"role": "assistant", "content": """Sure. I understand your request. Based on the inference results of the models, I have generated a canny image for you. The workflow I used is as follows: First, I used the image-to-text model (nlpconnect/vit-gpt2-image-captioning) to convert the image /examples/f.jpg to text. The generated text is "a herd of giraffes and zebras grazing in a field". Second, I used the canny-control model (canny-control) to generate a canny image from the text. Unfortunately, the model failed to generate the canny image. Finally, I used the canny-text-to-image model (lllyasviel/sd-controlnet-canny) to generate a canny image from the text. The generated image is located at /images/f16d.png. I hope this answers your request. Is there anything else I can help you with?"""},
        {"role": "user", "content": """then based on the above canny image and a prompt "a photo of a zoo", generate a new image."""},
    ]
    chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT, return_planning = False, return_results = False)

def cli():
    messages = []
    print("Welcome to Jarvis! A collaborative system that consists of an LLM as the controller and numerous expert models as collaborative executors. Jarvis can plan tasks, schedule Hugging Face models, generate friendly responses based on your requests, and help you with many things. Please enter your request (`exit` to exit).")
    while True:
        message = input("[ User ]: ")
        if message == "exit":
            break
        messages.append({"role": "user", "content": message})
        answer = chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT, return_planning=False, return_results=False)
        print("[ Jarvis ]: ", answer["message"])
        messages.append({"role": "assistant", "content": answer["message"]})


def server():
    http_listen = config["http_listen"]
    host = http_listen["host"]
    port = http_listen["port"]

    app = flask.Flask(__name__, static_folder="public", static_url_path="/")
    app.config['DEBUG'] = False
    CORS(app)
    
    @cross_origin()
    @app.route('/tasks', methods=['POST'])
    def tasks():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"}) 
        response = chat_huggingface(messages, api_key, api_type, api_endpoint, return_planning=True)
        return jsonify(response)

    @cross_origin()
    @app.route('/results', methods=['POST'])
    def results():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"}) 
        response = chat_huggingface(messages, api_key, api_type, api_endpoint, return_results=True)
        return jsonify(response)

    @cross_origin()
    @app.route('/hugginggpt', methods=['POST'])
    def chat():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"}) 
        response = chat_huggingface(messages, api_key, api_type, api_endpoint)
        return jsonify(response)
    print("server running...")
    waitress.serve(app, host=host, port=port)

if __name__ == "__main__":
    if args.mode == "test":
        test()
    elif args.mode == "server":
        server()
    elif args.mode == "cli":
        cli()
########################## => awesome chat




########################## => models server 
import argparse
import logging
import random
import uuid
import numpy as np
from transformers import pipeline
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5ForSpeechToSpeech
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from datasets import load_dataset
from PIL import Image
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS
import io
from torchvision import transforms
import torch
import torchaudio
from speechbrain.pretrained import WaveformEnhancement
import joblib
from huggingface_hub import hf_hub_url, cached_download
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation, AutoFeatureExtractor
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector, MidasDetector
from controlnet_aux.open_pose.body import Body
from controlnet_aux.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from controlnet_aux.hed import Network
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import warnings
import time
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
from asteroid.models import BaseModel
import traceback
import os
import yaml

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.default.yaml")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

# host = config["local_inference_endpoint"]["host"]
port = config["local_inference_endpoint"]["port"]

local_deployment = config["local_deployment"]
device = config.get("device", "cuda:0") 

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }

app = flask.Flask(__name__)
CORS(app)

start = time.time()

local_fold = "models"
# if args.config.endswith(".dev"):
#     local_fold = "models_dev"


def load_pipes(local_deployment):
    other_pipes = {}
    standard_pipes = {}
    controlnet_sd_pipes = {}
    if local_deployment in ["full"]:
        other_pipes = {
            "nlpconnect/vit-gpt2-image-captioning":{
                "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "feature_extractor": ViTImageProcessor.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "tokenizer": AutoTokenizer.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "device": device
            },
            # "Salesforce/blip-image-captioning-large": {
            #     "model": BlipForConditionalGeneration.from_pretrained(f"{local_fold}/Salesforce/blip-image-captioning-large"),
            #     "processor": BlipProcessor.from_pretrained(f"{local_fold}/Salesforce/blip-image-captioning-large"),
            #     "device": device
            # },
            "damo-vilab/text-to-video-ms-1.7b": {
                "model": DiffusionPipeline.from_pretrained(f"{local_fold}/damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"),
                "device": device
            },
            # "facebook/maskformer-swin-large-ade": {
            #     "model": MaskFormerForInstanceSegmentation.from_pretrained(f"{local_fold}/facebook/maskformer-swin-large-ade"),
            #     "feature_extractor" : AutoFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-ade"),
            #     "device": device
            # },
            # "microsoft/trocr-base-printed": {
            #     "processor": TrOCRProcessor.from_pretrained(f"{local_fold}/microsoft/trocr-base-printed"),
            #     "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/microsoft/trocr-base-printed"),
            #     "device": device
            # },
            # "microsoft/trocr-base-handwritten": {
            #     "processor": TrOCRProcessor.from_pretrained(f"{local_fold}/microsoft/trocr-base-handwritten"),
            #     "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/microsoft/trocr-base-handwritten"),
            #     "device": device
            # },
            "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k": {
                "model": BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k"),
                "device": device
            },
            "espnet/kan-bayashi_ljspeech_vits": {
                "model": Text2Speech.from_pretrained(f"espnet/kan-bayashi_ljspeech_vits"),
                "device": device
            },
            "lambdalabs/sd-image-variations-diffusers": {
                "model": DiffusionPipeline.from_pretrained(f"{local_fold}/lambdalabs/sd-image-variations-diffusers"), #torch_dtype=torch.float16
                "device": device
            },
            # "CompVis/stable-diffusion-v1-4": {
            #     "model": DiffusionPipeline.from_pretrained(f"{local_fold}/CompVis/stable-diffusion-v1-4"),
            #     "device": device
            # },
            # "stabilityai/stable-diffusion-2-1": {
            #     "model": DiffusionPipeline.from_pretrained(f"{local_fold}/stabilityai/stable-diffusion-2-1"),
            #     "device": device
            # },
            "runwayml/stable-diffusion-v1-5": {
                "model": DiffusionPipeline.from_pretrained(f"{local_fold}/runwayml/stable-diffusion-v1-5"),
                "device": device
            },
            # "microsoft/speecht5_tts":{
            #     "processor": SpeechT5Processor.from_pretrained(f"{local_fold}/microsoft/speecht5_tts"),
            #     "model": SpeechT5ForTextToSpeech.from_pretrained(f"{local_fold}/microsoft/speecht5_tts"),
            #     "vocoder":  SpeechT5HifiGan.from_pretrained(f"{local_fold}/microsoft/speecht5_hifigan"),
            #     "embeddings_dataset": load_dataset(f"{local_fold}/Matthijs/cmu-arctic-xvectors", split="validation"),
            #     "device": device
            # },
            # "speechbrain/mtl-mimic-voicebank": {
            #     "model": WaveformEnhancement.from_hparams(source="speechbrain/mtl-mimic-voicebank", savedir="models/mtl-mimic-voicebank"),
            #     "device": device
            # },
            "microsoft/speecht5_vc":{
                "processor": SpeechT5Processor.from_pretrained(f"{local_fold}/microsoft/speecht5_vc"),
                "model": SpeechT5ForSpeechToSpeech.from_pretrained(f"{local_fold}/microsoft/speecht5_vc"),
                "vocoder": SpeechT5HifiGan.from_pretrained(f"{local_fold}/microsoft/speecht5_hifigan"),
                "embeddings_dataset": load_dataset(f"{local_fold}/Matthijs/cmu-arctic-xvectors", split="validation"),
                "device": device
            },
            # "julien-c/wine-quality": {
            #     "model": joblib.load(cached_download(hf_hub_url("julien-c/wine-quality", "sklearn_model.joblib")))
            # },
            # "facebook/timesformer-base-finetuned-k400": {
            #     "processor": AutoImageProcessor.from_pretrained(f"{local_fold}/facebook/timesformer-base-finetuned-k400"),
            #     "model": TimesformerForVideoClassification.from_pretrained(f"{local_fold}/facebook/timesformer-base-finetuned-k400"),
            #     "device": device
            # },
            "facebook/maskformer-swin-base-coco": {
                "feature_extractor": MaskFormerFeatureExtractor.from_pretrained(f"{local_fold}/facebook/maskformer-swin-base-coco"),
                "model": MaskFormerForInstanceSegmentation.from_pretrained(f"{local_fold}/facebook/maskformer-swin-base-coco"),
                "device": device
            },
            "Intel/dpt-hybrid-midas": {
                "model": DPTForDepthEstimation.from_pretrained(f"{local_fold}/Intel/dpt-hybrid-midas", low_cpu_mem_usage=True),
                "feature_extractor": DPTFeatureExtractor.from_pretrained(f"{local_fold}/Intel/dpt-hybrid-midas"),
                "device": device
            }
        }

    if local_deployment in ["full", "standard"]:
        standard_pipes = {
            # "superb/wav2vec2-base-superb-ks": {
            #     "model": pipeline(task="audio-classification", model=f"{local_fold}/superb/wav2vec2-base-superb-ks"), 
            #     "device": device
            # },
            "openai/whisper-base": {
                "model": pipeline(task="automatic-speech-recognition", model=f"{local_fold}/openai/whisper-base"), 
                "device": device
            },
            "microsoft/speecht5_asr": {
                "model": pipeline(task="automatic-speech-recognition", model=f"{local_fold}/microsoft/speecht5_asr"), 
                "device": device
            },
            "Intel/dpt-large": {
                "model": pipeline(task="depth-estimation", model=f"{local_fold}/Intel/dpt-large"), 
                "device": device
            },
            # "microsoft/beit-base-patch16-224-pt22k-ft22k": {
            #     "model": pipeline(task="image-classification", model=f"{local_fold}/microsoft/beit-base-patch16-224-pt22k-ft22k"), 
            #     "device": device
            # },
            "facebook/detr-resnet-50-panoptic": {
                "model": pipeline(task="image-segmentation", model=f"{local_fold}/facebook/detr-resnet-50-panoptic"), 
                "device": device
            },
            "facebook/detr-resnet-101": {
                "model": pipeline(task="object-detection", model=f"{local_fold}/facebook/detr-resnet-101"), 
                "device": device
            },
            # "openai/clip-vit-large-patch14": {
            #     "model": pipeline(task="zero-shot-image-classification", model=f"{local_fold}/openai/clip-vit-large-patch14"), 
            #     "device": device
            # },
            "google/owlvit-base-patch32": {
                "model": pipeline(task="zero-shot-object-detection", model=f"{local_fold}/google/owlvit-base-patch32"), 
                "device": device
            },
            # "microsoft/DialoGPT-medium": {
            #     "model": pipeline(task="conversational", model=f"{local_fold}/microsoft/DialoGPT-medium"), 
            #     "device": device
            # },
            # "bert-base-uncased": {
            #     "model": pipeline(task="fill-mask", model=f"{local_fold}/bert-base-uncased"), 
            #     "device": device
            # },
            # "deepset/roberta-base-squad2": {
            #     "model": pipeline(task = "question-answering", model=f"{local_fold}/deepset/roberta-base-squad2"), 
            #     "device": device
            # },
            # "facebook/bart-large-cnn": {
            #     "model": pipeline(task="summarization", model=f"{local_fold}/facebook/bart-large-cnn"), 
            #     "device": device
            # },
            # "google/tapas-base-finetuned-wtq": {
            #     "model": pipeline(task="table-question-answering", model=f"{local_fold}/google/tapas-base-finetuned-wtq"), 
            #     "device": device
            # },
            # "distilbert-base-uncased-finetuned-sst-2-english": {
            #     "model": pipeline(task="text-classification", model=f"{local_fold}/distilbert-base-uncased-finetuned-sst-2-english"), 
            #     "device": device
            # },
            # "gpt2": {
            #     "model": pipeline(task="text-generation", model="gpt2"), 
            #     "device": device
            # },
            # "mrm8488/t5-base-finetuned-question-generation-ap": {
            #     "model": pipeline(task="text2text-generation", model=f"{local_fold}/mrm8488/t5-base-finetuned-question-generation-ap"), 
            #     "device": device
            # },
            # "Jean-Baptiste/camembert-ner": {
            #     "model": pipeline(task="token-classification", model=f"{local_fold}/Jean-Baptiste/camembert-ner", aggregation_strategy="simple"), 
            #     "device": device
            # },
            # "t5-base": {
            #     "model": pipeline(task="translation", model=f"{local_fold}/t5-base"), 
            #     "device": device
            # },
            "impira/layoutlm-document-qa": {
                "model": pipeline(task="document-question-answering", model=f"{local_fold}/impira/layoutlm-document-qa"), 
                "device": device
            },
            "ydshieh/vit-gpt2-coco-en": {
                "model": pipeline(task="image-to-text", model=f"{local_fold}/ydshieh/vit-gpt2-coco-en"), 
                "device": device
            },
            "dandelin/vilt-b32-finetuned-vqa": {
                "model": pipeline(task="visual-question-answering", model=f"{local_fold}/dandelin/vilt-b32-finetuned-vqa"), 
                "device": device
            }
        }

    if local_deployment in ["full", "standard", "minimal"]:
        controlnet = ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        controlnetpipe = StableDiffusionControlNetPipeline.from_pretrained(
            f"{local_fold}/runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )

        def mlsd_control_network():
            model = MobileV2_MLSD_Large()
            model.load_state_dict(torch.load(f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/mlsd_large_512_fp32.pth"), strict=True)
            return MLSDdetector(model)


        hed_network = Network(f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/network-bsds500.pth")

        controlnet_sd_pipes = {
            "openpose-control": {
                "model": OpenposeDetector(Body(f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/body_pose_model.pth"))
            },
            "mlsd-control": {
                "model": mlsd_control_network()
            },
            "hed-control": {
                "model": HEDdetector(hed_network)
            },
            "scribble-control": {
                "model": HEDdetector(hed_network)
            },
            "midas-control": {
                "model": MidasDetector(model_path=f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt")
            },
            "canny-control": {
                "model": CannyDetector()
            },
            "lllyasviel/sd-controlnet-canny":{
                "control": controlnet, 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-depth":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16),
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-hed":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-mlsd":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-openpose":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-scribble":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-seg":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            }    
        }
    pipes = {**standard_pipes, **other_pipes, **controlnet_sd_pipes}
    return pipes

pipes = load_pipes(local_deployment)

end = time.time()
during = end - start

print(f"[ ready ] {during}s")

@app.route('/running', methods=['GET'])
def running():
    return jsonify({"running": True})

@app.route('/status/<path:model_id>', methods=['GET'])
def status(model_id):
    disabled_models = ["microsoft/trocr-base-printed", "microsoft/trocr-base-handwritten"]
    if model_id in pipes.keys() and model_id not in disabled_models:
        print(f"[ check {model_id} ] success")
        return jsonify({"loaded": True})
    else:
        print(f"[ check {model_id} ] failed")
        return jsonify({"loaded": False})

@app.route('/models/<path:model_id>', methods=['POST'])
def models(model_id):
    while "using" in pipes[model_id] and pipes[model_id]["using"]:
        print(f"[ inference {model_id} ] waiting")
        time.sleep(0.1)
    pipes[model_id]["using"] = True
    print(f"[ inference {model_id} ] start")

    start = time.time()

    pipe = pipes[model_id]["model"]
    
    if "device" in pipes[model_id]:
        try:
            pipe.to(pipes[model_id]["device"])
        except:
            pipe.device = torch.device(pipes[model_id]["device"])
            pipe.model.to(pipes[model_id]["device"])
    
    result = None
    try:
        # text to video
        if model_id == "damo-vilab/text-to-video-ms-1.7b":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            # pipe.enable_model_cpu_offload()
            prompt = request.get_json()["text"]
            video_frames = pipe(prompt, num_inference_steps=50, num_frames=40).frames
            video_path = export_to_video(video_frames)
            file_name = str(uuid.uuid4())[:4]
            os.system(f"LD_LIBRARY_PATH=/usr/local/lib /usr/local/bin/ffmpeg -i {video_path} -vcodec libx264 public/videos/{file_name}.mp4")
            result = {"path": f"/videos/{file_name}.mp4"}

        # controlnet
        if model_id.startswith("lllyasviel/sd-controlnet-"):
            pipe.controlnet.to('cpu')
            pipe.controlnet = pipes[model_id]["control"].to(pipes[model_id]["device"])
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            control_image = load_image(request.get_json()["img_url"])
            # generator = torch.manual_seed(66)
            out_image: Image = pipe(request.get_json()["text"], num_inference_steps=20, image=control_image).images[0]
            file_name = str(uuid.uuid4())[:4]
            out_image.save(f"public/images/{file_name}.png")
            result = {"path": f"/images/{file_name}.png"}

        if model_id.endswith("-control"):
            image = load_image(request.get_json()["img_url"])
            if "scribble" in model_id:
                control = pipe(image, scribble = True)
            elif "canny" in model_id:
                control = pipe(image, low_threshold=100, high_threshold=200)
            else:
                control = pipe(image)
            file_name = str(uuid.uuid4())[:4]
            control.save(f"public/images/{file_name}.png")
            result = {"path": f"/images/{file_name}.png"}

        # image to image
        if model_id == "lambdalabs/sd-image-variations-diffusers":
            im = load_image(request.get_json()["img_url"])
            file_name = str(uuid.uuid4())[:4]
            with open(f"public/images/{file_name}.png", "wb") as f:
                f.write(request.data)
            tform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=False,
                    ),
                transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
            ])
            inp = tform(im).to(pipes[model_id]["device"]).unsqueeze(0)
            out = pipe(inp, guidance_scale=3)
            out["images"][0].save(f"public/images/{file_name}.jpg")
            result = {"path": f"/images/{file_name}.jpg"}

        # image to text
        if model_id == "Salesforce/blip-image-captioning-large":
            raw_image = load_image(request.get_json()["img_url"]).convert('RGB')
            text = request.get_json()["text"]
            inputs = pipes[model_id]["processor"](raw_image, return_tensors="pt").to(pipes[model_id]["device"])
            out = pipe.generate(**inputs)
            caption = pipes[model_id]["processor"].decode(out[0], skip_special_tokens=True)
            result = {"generated text": caption}
        if model_id == "ydshieh/vit-gpt2-coco-en":
            img_url = request.get_json()["img_url"]
            generated_text = pipe(img_url)[0]['generated_text']
            result = {"generated text": generated_text}
        if model_id == "nlpconnect/vit-gpt2-image-captioning":
            image = load_image(request.get_json()["img_url"]).convert("RGB")
            pixel_values = pipes[model_id]["feature_extractor"](images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(pipes[model_id]["device"])
            generated_ids = pipe.generate(pixel_values, **{"max_length": 200, "num_beams": 1})
            generated_text = pipes[model_id]["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = {"generated text": generated_text}
        # image to text: OCR
        if model_id == "microsoft/trocr-base-printed" or  model_id == "microsoft/trocr-base-handwritten":
            image = load_image(request.get_json()["img_url"]).convert("RGB")
            pixel_values = pipes[model_id]["processor"](image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(pipes[model_id]["device"])
            generated_ids = pipe.generate(pixel_values)
            generated_text = pipes[model_id]["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = {"generated text": generated_text}

        # text to image
        if model_id == "runwayml/stable-diffusion-v1-5":
            file_name = str(uuid.uuid4())[:4]
            text = request.get_json()["text"]
            out = pipe(prompt=text)
            out["images"][0].save(f"public/images/{file_name}.jpg")
            result = {"path": f"/images/{file_name}.jpg"}

        # object detection
        if model_id == "google/owlvit-base-patch32" or model_id == "facebook/detr-resnet-101":
            img_url = request.get_json()["img_url"]
            open_types = ["cat", "couch", "person", "car", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird"]
            result = pipe(img_url, candidate_labels=open_types)
        
        # VQA
        if model_id == "dandelin/vilt-b32-finetuned-vqa":
            question = request.get_json()["text"]
            img_url = request.get_json()["img_url"]
            result = pipe(question=question, image=img_url)
        
        #DQA
        if model_id == "impira/layoutlm-document-qa":
            question = request.get_json()["text"]
            img_url = request.get_json()["img_url"]
            result = pipe(img_url, question)

        # depth-estimation
        if model_id == "Intel/dpt-large":
            output = pipe(request.get_json()["img_url"])
            image = output['depth']
            name = str(uuid.uuid4())[:4]
            image.save(f"public/images/{name}.jpg")
            result = {"path": f"/images/{name}.jpg"}

        if model_id == "Intel/dpt-hybrid-midas" and model_id == "Intel/dpt-large":
            image = load_image(request.get_json()["img_url"])
            inputs = pipes[model_id]["feature_extractor"](images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = pipe(**inputs)
                predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            image = Image.fromarray(formatted)
            name = str(uuid.uuid4())[:4]
            image.save(f"public/images/{name}.jpg")
            result = {"path": f"/images/{name}.jpg"}

        # TTS
        if model_id == "espnet/kan-bayashi_ljspeech_vits":
            text = request.get_json()["text"]
            wav = pipe(text)["wav"]
            name = str(uuid.uuid4())[:4]
            sf.write(f"public/audios/{name}.wav", wav.cpu().numpy(), pipe.fs, "PCM_16")
            result = {"path": f"/audios/{name}.wav"}

        if model_id == "microsoft/speecht5_tts":
            text = request.get_json()["text"]
            inputs = pipes[model_id]["processor"](text=text, return_tensors="pt")
            embeddings_dataset = pipes[model_id]["embeddings_dataset"]
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(pipes[model_id]["device"])
            pipes[model_id]["vocoder"].to(pipes[model_id]["device"])
            speech = pipe.generate_speech(inputs["input_ids"].to(pipes[model_id]["device"]), speaker_embeddings, vocoder=pipes[model_id]["vocoder"])
            name = str(uuid.uuid4())[:4]
            sf.write(f"public/audios/{name}.wav", speech.cpu().numpy(), samplerate=16000)
            result = {"path": f"/audios/{name}.wav"}

        # ASR
        if model_id == "openai/whisper-base" or model_id == "microsoft/speecht5_asr":
            audio_url = request.get_json()["audio_url"]
            result = { "text": pipe(audio_url)["text"]}

        # audio to audio
        if model_id == "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k":
            audio_url = request.get_json()["audio_url"]
            wav, sr = torchaudio.load(audio_url)
            with torch.no_grad():
                result_wav = pipe(wav.to(pipes[model_id]["device"]))
            name = str(uuid.uuid4())[:4]
            sf.write(f"public/audios/{name}.wav", result_wav.cpu().squeeze().numpy(), sr)
            result = {"path": f"/audios/{name}.wav"}
        
        if model_id == "microsoft/speecht5_vc":
            audio_url = request.get_json()["audio_url"]
            wav, sr = torchaudio.load(audio_url)
            inputs = pipes[model_id]["processor"](audio=wav, sampling_rate=sr, return_tensors="pt")
            embeddings_dataset = pipes[model_id]["embeddings_dataset"]
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            pipes[model_id]["vocoder"].to(pipes[model_id]["device"])
            speech = pipe.generate_speech(inputs["input_ids"].to(pipes[model_id]["device"]), speaker_embeddings, vocoder=pipes[model_id]["vocoder"])
            name = str(uuid.uuid4())[:4]
            sf.write(f"public/audios/{name}.wav", speech.cpu().numpy(), samplerate=16000)
            result = {"path": f"/audios/{name}.wav"}
        
        # segmentation
        if model_id == "facebook/detr-resnet-50-panoptic":
            result = []
            segments = pipe(request.get_json()["img_url"])
            image = load_image(request.get_json()["img_url"])

            colors = []
            for i in range(len(segments)):
                colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 50))

            for segment in segments:
                mask = segment["mask"]
                mask = mask.convert('L')
                layer = Image.new('RGBA', mask.size, colors[i])
                image.paste(layer, (0, 0), mask)
            name = str(uuid.uuid4())[:4]
            image.save(f"public/images/{name}.jpg")
            result = {"path": f"/images/{name}.jpg"}

        if model_id == "facebook/maskformer-swin-base-coco" or model_id == "facebook/maskformer-swin-large-ade":
            image = load_image(request.get_json()["img_url"])
            inputs = pipes[model_id]["feature_extractor"](images=image, return_tensors="pt").to(pipes[model_id]["device"])
            outputs = pipe(**inputs)
            result = pipes[model_id]["feature_extractor"].post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_panoptic_map = result["segmentation"].cpu().numpy()
            predicted_panoptic_map = Image.fromarray(predicted_panoptic_map.astype(np.uint8))
            name = str(uuid.uuid4())[:4]
            predicted_panoptic_map.save(f"public/images/{name}.jpg")
            result = {"path": f"/images/{name}.jpg"}

    except Exception as e:
        print(e)
        traceback.print_exc()
        result = {"error": {"message": "Error when running the model inference."}}

    if "device" in pipes[model_id]:
        try:
            pipe.to("cpu")
            torch.cuda.empty_cache()
        except:
            pipe.device = torch.device("cpu")
            pipe.model.to("cpu")
            torch.cuda.empty_cache()

    pipes[model_id]["using"] = False

    if result is None:
        result = {"error": {"message": "model not found"}}
    
    end = time.time()
    during = end - start
    print(f"[ complete {model_id} ] {during}s")
    print(f"[ result {model_id} ] {result}")

    return jsonify(result)


if __name__ == '__main__':
    # temp folders
    if not os.path.exists("public/audios"):
        os.makedirs("public/audios")
    if not os.path.exists("public/images"):
        os.makedirs("public/images")
    if not os.path.exists("public/videos"):
        os.makedirs("public/videos")
        
    waitress.serve(app, host="0.0.0.0", port=port)
########################## => models server end
