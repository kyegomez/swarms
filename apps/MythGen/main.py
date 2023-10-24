import openai
import os
import dotenv
import logging
import gradio as gr
from dalle3 import Dalle
from bing_chat import BingChat

# from swarms.models.bingchat import BingChat  

dotenv.load_dotenv(".env")

# Initialize the EdgeGPTModel
openai_api_key = os.getenv("OPENAI_API_KEY")
model = BingChat(cookie_path = "./cookies.json")
cookie = os.environ.get("BING_COOKIE")
auth = os.environ.get("AUTH_COOKIE")



response = model("Generate")

# Initialize DALLE3 API
cookie = os.getenv("DALLE_COOKIE")
dalle = Dalle(cookie)

logging.basicConfig(level=logging.INFO)

accumulated_story = ""
latest_caption = ""
standard_suffix = ""
storyboard = []

def generate_images_with_dalle(caption):
    model.create_img(auth_cookie=cookie,auth_cookie_SRCHHPGUSR=auth,prompt=caption)
    urls = dalle.get_urls()
    return urls

def generate_single_caption(text):
    prompt = f"A comic about {text}."
    response = model(prompt)
    return response

def interpret_text_with_gpt(text, suffix):
    return generate_single_caption(f"{text} {suffix}")

def create_standard_suffix(original_prompt):
    return f"In the style of {original_prompt}"

def gradio_interface(text=None, next_button_clicked=False):
    global accumulated_story, latest_caption, standard_suffix, storyboard
    
    if not standard_suffix:
        standard_suffix = create_standard_suffix(text)
    
    if next_button_clicked:
        new_caption = interpret_text_with_gpt(latest_caption, standard_suffix)
        new_urls = generate_images_with_dalle(new_caption)
        latest_caption = new_caption
        storyboard.append((new_urls, new_caption))
        
    elif text:
        caption = interpret_text_with_gpt(text, standard_suffix)
        comic_panel_urls = generate_images_with_dalle(caption)
        latest_caption = caption
        storyboard.append((comic_panel_urls, caption))

    storyboard_html = ""
    for urls, cap in storyboard:
        for url in urls:
            storyboard_html += f'<img src="{url}" alt="{cap}" width="300"/><br>{cap}<br>'

    return storyboard_html

if __name__ == "__main__":
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.inputs.Textbox(default="Type your story concept here", optional=True, label="Story Concept"),
            gr.inputs.Checkbox(label="Generate Next Part")
        ],
        outputs=[gr.outputs.HTML()],
        live=False  # Submit button will appear
    )
    iface.launch()
