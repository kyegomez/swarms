import requests
import gradio as gr

def api_response(message, history):
    # Extract the string content from the Gradio message
    user_message = message["content"] if isinstance(message, dict) else message
    
    url = "http://localhost:8888/chat"
    payload = {
        "id": "string",
        "model": {
            "id": "llama-2-70b.Q5_K_M",
            "name": "llama-2-70b.Q5_K_M",
            "maxLength": 2048,
            "tokenLimit": 2048
        },
        "messages": [
            {
            "role": "system",
            "content": "Hello, how may I help you?  AMA!"
            },
            {
            "role": "user",
            "content": user_message  # Use the extracted message content here
            }
        ],
        "maxTokens": 2048,
        "temperature": 0,
        "prompt": "HUMAN: \n You are a helpful AI assistant.  Use the following context and chat history to answer the question at the end with a helpful answer.  Get straight to the point and always think things through step-by-step before answering.  If you don't know the answer, just say 'I don't know'; don't try to make up an answer. \n\n<context>{context}</context>\n<chat_history>{chat_history}</chat_history>\n<question>{question}</question>\n\nAI:  Here is the most relevant sentence in the context:  \n",
        "file": {
            "filename": "None",
            "title": "None",
            "username": "None",
            "state": "Unavailable"
        }
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("answer", "Error: No answer returned")
    else:
        return f"Error: {response.status_code}"

gr.ChatInterface(api_response).launch()