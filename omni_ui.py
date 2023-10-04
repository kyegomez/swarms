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
    chat_history.append(agent_response)
    
    return render_chat(chat_history)

def render_chat(chat_history):
    chat_str = '<div style="overflow-y: scroll; height: 400px;">'
    for message in chat_history:
        timestamp = message.get('timestamp', 'N/A')
        if message['type'] == 'user':
            chat_str += f'<div style="text-align: right; color: blue; margin: 5px; border-radius: 10px; background-color: #E0F0FF; padding: 5px;">{message["content"]}<br><small>{timestamp}</small></div>'
        elif message['type'] == 'text':
            chat_str += f'<div style="text-align: left; color: green; margin: 5px; border-radius: 10px; background-color: #E0FFE0; padding: 5px;">{message["content"]}<br><small>{timestamp}</small></div>'
        elif message['type'] == 'image':
            img_path = os.path.join("root_directory", message['content'])
            chat_str += f'<div style="text-align: left; margin: 5px;"><img src="{img_path}" alt="image" style="max-width: 100%; border-radius: 10px;"/><br><small>{timestamp}</small></div>'
    chat_str += '</div>'
    return chat_str

# Define Gradio interface
iface = Interface(
    fn=update_chat, 
    inputs=gr.inputs.Textbox(lines=2, placeholder="Type your message here..."),
    outputs=gr.outputs.HTML(label="Chat History"),
    live=True,
    title="Conversational AI Interface",
    description="Chat with our AI agent!",
    allow_flagging=False
)

iface.launch()
