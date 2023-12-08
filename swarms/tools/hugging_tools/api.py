from functools import wraps
import io
import json
import os
import random
import requests
from ..tool import Tool
from huggingface_hub.inference_api import InferenceApi
import base64
from io import BytesIO
import os
import random
import uuid
import requests
from PIL import Image, ImageDraw
from diffusers.utils import load_image
from pydub import AudioSegment
from huggingface_hub.inference_api import InferenceApi
from huggingface_hub.utils._errors import RepositoryNotFoundError

DIRPATH = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "debug": False,
    "log_file": None,
    "huggingface": {"token": os.environ.get("HUGGINGFACE_API_KEY")},
    "proxy": None,
    "inference_mode": "huggingface",
    "local_inference_endpoint": {"host": "localhost", "port": 8005},
    "huggingface_inference_endpoint": {
        "host": "api-inference.huggingface.co",
        "port": 443,
    },
}

HUGGINGFACE_HEADERS = {}
if CONFIG["huggingface"]["token"] and CONFIG["huggingface"]["token"].startswith(
    "hf_"
):  # Check for valid huggingface token in config file
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {CONFIG['huggingface']['token']}",
    }
elif "HUGGINGFACE_ACCESS_TOKEN" in os.environ and os.getenv(
    "HUGGINGFACE_API_KEY"
).startswith(
    "hf_"
):  # Check for environment variable HUGGINGFACE_ACCESS_TOKEN
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
    }
else:
    raise ValueError(f"Incrorrect HuggingFace token. Please check your file.")

PROXY = None
if CONFIG["proxy"]:
    PROXY = {
        "https": CONFIG["proxy"],
    }

INFERENCE_MODE = CONFIG["inference_mode"]

DOCS_PATH = "sources/docs.json"

INPUT_PATH = "files"
OUTPUT_PATH = "files"

MODEL_SERVER = None
if INFERENCE_MODE != "huggingface":
    MODEL_SERVER = (
        "http://"
        + CONFIG["local_inference_endpoint"]["host"]
        + ":"
        + str(CONFIG["local_inference_endpoint"]["port"])
    )
    message = f"The server of local inference endpoints is not running, please start it first. (or using `inference_mode: huggingface` in for a feature-limited experience)"
    try:
        r = requests.get(MODEL_SERVER + "/running")
        if r.status_code != 200:
            raise ValueError(message)
    except:
        raise ValueError(message)


def get_model_status(model_id, url, headers):
    # endpoint_type = "huggingface" if "huggingface" in url else "local"
    if "huggingface" in url:
        r = requests.get(url + f"/{model_id}", headers=headers, proxies=PROXY)
    else:
        r = requests.get(url)
    return r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]


def image_to_bytes(img_url):
    img_byte = io.BytesIO()
    load_image(img_url).save(img_byte, format="png")
    img_data = img_byte.getvalue()
    return img_data


def build_tool(conf) -> Tool:
    task_list = []
    tool = Tool(
        tool_name="hugging_tools",
        description="API interface for HuggingGPT-like applications.",
        name_for_model="hugging_tools",
        description_for_model="""This API interface provides easy access to popular models available on the Huggingface model hub. You MUST check model_docs to fetch the available models FIRST: 
            Action: model_docs
            Action Input: {"task" : <task_name>}
            After that you can choose an available models. """,
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="test@123.com",
        legal_info_url="hello@legal.com",
    )

    # set the get route to /func.__name__ and format docs
    def task(func):
        func.__doc__ = (
            """You MUST check model_docs to fetch the available models FIRST: 
            Action: model_docs
            Action Input: {"task" : "%s"}
            After that you can choose an available models in the list. 
        """
            % func.__name__
        )

        @wraps(func)
        def try_run_task(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RepositoryNotFoundError as e:
                return """The model with model_id you input is not available. Plese check the model_docs to get other available models:
                    Action: model_docs
                    Action Input: {"task" : "%s"}
                    After that you can choose an available models in the list. 
                """

        path = "/" + func.__name__
        try_run_task.route = path
        task_list.append(func.__name__)
        return tool.get(path)(try_run_task)

    def format_docs(str):
        def set_docs(func):
            func.__doc__ = func.__doc__ % str

            @wraps(func)
            def original_func(*args, **kwargs):
                return func(*args, **kwargs)

            return original_func

        return set_docs

    @task
    def question_answering(model_id: str, question: str, context: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(
            inference({"question": question, "context": (context if context else "")})
        )

    @task
    def sentence_similarity(model_id: str, text: str, context: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(
            inference(
                {"source_sentence": text, "sentences": [(context if context else "")]}
            )
        )

    @task
    def text_classification(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(inference(text))

    @task
    def token_classification(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(inference(text))

    @task
    def text2text_generation(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(inference(text))

    @task
    def summarization(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(inference(text))

    @task
    def translation(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(inference(text))

    @task
    def conversational(
        model_id: str, text: str, past_user_inputs: str, generated_responses: str
    ) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        input = {
            "past_user_inputs": [past_user_inputs],
            "generated_responses": [generated_responses],
            "text": text,
        }
        return str(inference(input))

    @task
    def text_generation(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        return str(inference(text))

    # CV tasks
    @task
    def visual_question_answering(
        model_id: str, image_file_name: str, text: str
    ) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        return str(inference({"question": text, "image": img_base64}))

    @task
    def document_question_answering(
        model_id: str, image_file_name: str, text: str
    ) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        return str(inference({"question": text, "image": img_base64}))

    @task
    def image_to_image(model_id: str, image_file_name: str) -> str:
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        # result = inference(data=img_data) # not support
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        task_url = f"https://api-inference.huggingface.co/models/{model_id}"
        r = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data)
        name = str(uuid.uuid4())[:4]
        result = f"{name}.jpeg"
        with open(f"{DIRPATH}/{OUTPUT_PATH}/{result}", "wb") as f:
            f.write(r.content)
        return result

    @task
    def text_to_image(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img = inference(text)
        name = str(uuid.uuid4())[:4]
        print(img.format)
        image_type = "jpg" if img.format == "JPEG" else "png"
        img.save(f"{DIRPATH}/{OUTPUT_PATH}/{name}.{image_type}")
        result = f"{name}.{image_type}"
        return result

    @task
    def image_segmentation(model_id: str, image_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        image = Image.open(BytesIO(img_data))
        predicted = inference(data=img_data)
        colors = []
        for i in range(len(predicted)):
            colors.append(
                (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                    155,
                )
            )
        for i, pred in enumerate(predicted):
            mask = pred.pop("mask").encode("utf-8")
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode="r")
            mask = mask.convert("L")

            layer = Image.new("RGBA", mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
        name = str(uuid.uuid4())[:4]
        image.save(f"{DIRPATH}/{OUTPUT_PATH}/{name}.jpg")
        result = {}
        result["generated image"] = f"{name}.jpg"
        result["predicted"] = predicted
        return str(result)

    @task
    def object_detection(model_id: str, image_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        predicted = inference(data=img_data)
        image = Image.open(BytesIO(img_data))
        draw = ImageDraw.Draw(image)
        labels = list(item["label"] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (
                    random.randint(0, 255),
                    random.randint(0, 100),
                    random.randint(0, 255),
                )
        for label in predicted:
            box = label["box"]
            draw.rectangle(
                ((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])),
                outline=color_map[label["label"]],
                width=2,
            )
            draw.text(
                (box["xmin"] + 5, box["ymin"] - 15),
                label["label"],
                fill=color_map[label["label"]],
            )
        name = str(uuid.uuid4())[:4]
        image.save(f"{DIRPATH}/{OUTPUT_PATH}/{name}.jpg")
        result = {}
        result["generated image"] = f"{name}.jpg"
        result["predicted"] = predicted
        return str(result)

    @task
    def image_classification(model_id: str, image_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        result = inference(data=img_data)
        return str(result)

    @task
    def image_to_text(model_id: str, image_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        img_data = image_to_bytes(f"{DIRPATH}/{INPUT_PATH}/{image_file_name}")
        result = inference(data=img_data)
        return str(result)

    # AUDIO tasks
    @task
    def text_to_speech(model_id: str, text: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        response = inference(text, raw_response=True)
        name = str(uuid.uuid4())[:4]
        with open(f"{DIRPATH}/{OUTPUT_PATH}/{name}.flac", "wb") as f:
            f.write(response.content)
        result = f"{name}.flac"
        return result

    @task
    def automatic_speech_recognition(model_id: str, audio_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        with open(f"{DIRPATH}/{INPUT_PATH}/{audio_file_name}", "rb") as f:
            audio = f.read()
            text = inference(data=audio, raw_response=True)
        result = text.content
        return str(result)

    @task
    def audio_to_audio(model_id: str, audio_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        with open(f"{DIRPATH}/{INPUT_PATH}/{audio_file_name}", "rb") as f:
            audio = f.read()
            response = inference(data=audio, raw_response=True)
        result = response.json()
        content = None
        type = None
        for k, v in result[0].items():
            if k == "blob":
                content = base64.b64decode(v.encode("utf-8"))
            if k == "content-type":
                type = "audio/flac".split("/")[-1]
        audio = AudioSegment.from_file(BytesIO(content))
        name = str(uuid.uuid4())[:4]
        audio.export(f"{DIRPATH}/{OUTPUT_PATH}/{name}.{type}", format=type)
        result = f"{name}.{type}"
        return result

    @task
    def audio_classification(model_id: str, audio_file_name: str) -> str:
        inference = InferenceApi(repo_id=model_id, token=CONFIG["huggingface"]["token"])
        with open(f"{DIRPATH}/{INPUT_PATH}/{audio_file_name}", "rb") as f:
            audio = f.read()
            response = inference(data=audio, raw_response=True)
        result = response.json()
        return str(result)

    @tool.get("/model_docs")
    @format_docs(str(task_list))
    def model_docs(task: str) -> str:
        """returns a document about the usage, examples, and available models of existing API.

        Every time before you use the API, you should get the document of EXISTING API ONLY and ONLY use the available models in the document.

        task: the name of API, includes %s

        return the document of the API.

        example:
        Action: model_docs
        Action Input: {"task" : "question_answering"}
        Observation: "question_answering is a function that uses a pre-trained language model to answer questions based on a given context.\n        You can choose one model from: 'distilbert-base-uncased-distilled-squad', 'deepset/minilm-uncased-squad2', 'etalab-ia/camembert-base-squadFR-fquad-piaf'.\n\n        Action Input: {\"model_id\" : \"distilbert-base-uncased-distilled-squad\", \"question\" : \"When did the first moon landing occur?\", \"context\" : \"The first manned moon landing was achieved by the United States on July 20, 1969, in the Apollo 11 mission.\"}\n        "
        """
        with open(f"{DIRPATH}/{DOCS_PATH}", "r") as f:
            docs = json.load(f)
        if task in docs.keys():
            return docs[task]
        else:
            return "The function doesn't exist. Please input the valid function name."

    return tool
