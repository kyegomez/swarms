import openai
import os
import dotenv
import logging
import gradio as gr
from BingImageCreator import ImageGen
from swarms.models.bing_chat import BingChat

# from swarms.models.bingchat import BingChat  
dotenv.load_dotenv(".env")

# Initialize the EdgeGPTModel
cookie = os.environ.get("BING_COOKIE")
auth = os.environ.get("AUTH_COOKIE")
model = BingChat(cookies_path="./cookies.json", bing_cookie="BING_COOKIE",auth_cookie="AUTH_COOKIE")

response = model("Generate")

logging.basicConfig(level=logging.INFO)

accumulated_story = ""
latest_caption = ""
standard_suffix = ""
storyboard = []

def generate_images_with_bingchat(caption):
    img_path = model.create_img(caption)
    img_urls = model.images(caption)
    return img_urls

def generate_single_caption(text):
    prompt = f"A comic about {text}."
    response = model(text)
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
        new_caption = generate_single_caption(latest_caption + " " + standard_suffix)
        new_urls = generate_images_with_bingchat(new_caption)
        latest_caption = new_caption
        storyboard.append((new_urls, new_caption))
        
    elif text:
        caption = generate_single_caption(text + " " + standard_suffix)
        comic_panel_urls = generate_images_with_bingchat(caption)
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
