# Import required libraries
from gradio import Interface, Textbox, HTML
import threading
import os
import glob
import base64
from swarms.models import OpenAIChat
from swarms.agents import OmniModalAgent


# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# Function to get the most recently created image in the directory
def get_latest_image():
    list_of_files = glob.glob("./*.png")  # Replace with your image file type
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# Initialize your OmniModalAgent
llm = OpenAIChat(model_name="gpt-4")  # Replace with your actual initialization
agent = OmniModalAgent(llm)  # Replace with your actual initialization

# Global variable to store chat history
chat_history = []


# Function to update chat
def update_chat(user_input):
    global chat_history
    chat_history.append({"type": "user", "content": user_input})

    # Get agent response
    agent_response = agent.run(user_input)

    # Handle the case where agent_response is not in the expected dictionary format
    if not isinstance(agent_response, dict):
        agent_response = {"type": "text", "content": str(agent_response)}

    chat_history.append(agent_response)

    # Check for the most recently created image and add it to the chat history
    latest_image = get_latest_image()
    if latest_image:
        chat_history.append({"type": "image", "content": latest_image})

    return render_chat(chat_history)


# Function to render chat as HTML


def render_chat(chat_history):
    chat_str = "<div style='max-height:400px;overflow-y:scroll;'>"
    for message in chat_history:
        if message["type"] == "user":
            chat_str += f"<p><strong>User:</strong> {message['content']}</p>"
        elif message["type"] == "text":
            chat_str += f"<p><strong>Agent:</strong> {message['content']}</p>"
        elif message["type"] == "image":
            img_path = os.path.join(".", message["content"])
            base64_img = image_to_base64(img_path)
            chat_str += f"<p><strong>Agent:</strong> <img src='data:image/png;base64,{base64_img}' alt='image' width='200'/></p>"
    chat_str += "</div>"
    return chat_str


# Define Gradio interface
iface = Interface(
    fn=update_chat,
    inputs=Textbox(label="Your Message", type="text"),
    outputs=HTML(label="Chat History"),
    live=True,
)


# Function to update the chat display
def update_display():
    global chat_history
    while True:
        iface.update(render_chat(chat_history))


# Run the update_display function in a separate thread
threading.Thread(target=update_display).start()

# Run Gradio interface
iface.launch()
