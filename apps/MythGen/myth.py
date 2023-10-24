from flask import Flask, request, jsonify
import openai
import logging
from dalle3 import Dalle
import os
import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import dotenv

app = Flask(__name__)

dotenv.load_dotenv(".env")

# Initialize OpenAI API, INPUT YOUR OWN OPENAI KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize DALLE3 API, INPUT YOUR OWN COOKIE
cookie = os.getenv("DALLE_COOKIE")
dalle = Dalle(cookie)


def interpret_text_with_gpt(text):
    model_engine = "text-davinci-002"
    panel_instructions = "Create a comic panel where"
    refined_prompt = f"{panel_instructions} {text}"
    
    response = openai.Completion.create(
        engine=model_engine,
        prompt=refined_prompt,
        max_tokens=100
    )
    
    final_prompt = response.choices[0].text.strip()
    return final_prompt

def generate_images_with_dalle(refined_prompt):
    dalle.create(refined_prompt)
    urls = dalle.get_urls()
    return urls

def gradio_interface(text):
    refined_prompt = interpret_text_with_gpt(text)
    comic_panel_urls = generate_images_with_dalle(refined_prompt)
    
    output = []
    for i, url in enumerate(comic_panel_urls):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        caption = f"Caption for panel {i+1}"
        output.append((img, caption))
        
    return output

iface = gr.Interface(
    fn=gradio_interface,
    inputs=["text"],
    outputs=[gr.outputs.Image(type="pil", label="Comic Panels"), "text"]
)

iface.launch()
