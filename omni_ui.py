import gradio as gr
from gradio import Interface
import threading
import os
from langchain.llms import OpenAIChat
from swarms.agents import OmniModalAgent

# Initialize the OmniModalAgent
llm = OpenAIChat(model_name="gpt-4")
agent = OmniModalAgent(llm)

# Global variable to store chat history
chat_history = []

def update_chat(user_input):
    global chat_history
    chat_history.append({"type": "user", "content": user_input})
    
    # Get agent response
    agent_response = agent.run(user_input)
    
    # Let's assume agent_response is a dictionary containing type and content.
    chat_history.append(agent_response)
    
    return render_chat(chat_history)

def render_chat(chat_history):
    chat_str = ""
    for message in chat_history:
        if message['type'] == 'user':
            chat_str += f"User: {message['content']}<br>"
        elif message['type'] == 'text':
            chat_str += f"Agent: {message['content']}<br>"
        elif message['type'] == 'image':
            img_path = os.path.join("root_directory", message['content'])
            chat_str += f"Agent: <img src='{img_path}' alt='image'/><br>"
    return chat_str

# Define Gradio interface
iface = Interface(
    fn=update_chat, 
    inputs="text", 
    outputs=gr.outputs.HTML(label="Chat History"),
    live=True
)

# Launch the Gradio interface
iface.launch()
