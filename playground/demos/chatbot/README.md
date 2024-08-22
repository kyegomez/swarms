# Swarms Chatbot Agent

## Installation

The Swarms Chatbot Agent is separated into 2 major parts, the [API server](#chatbot-api-server-fastapi) and the [UI](#chatbot-ui-nextjs).

## Requirements

### Using Local LLMs using vLLM

* Linux for vLLM.  Although this example can support other LLM hosts, we use vLLM for scalable distributed inference so the chatbot can scale up and out across GPUs and multiple servers using Ray, an optional dependency.  

* An NVidia GPU is expected but this example can run on CPU only if configured to do so.  This demo currently expects vLLM which requires Linux or WSL to run.

### Using OpenAI ChatGPT

In theory, any OpenAI compatible LLM endpoint is supported via the OpenAIChatLLM wrapper.

### Quickstart

* Start vLLM using GPUs using Docker container by running the [dockerRunVllm](./server/dockerRunVllm.sh).  Adjust the script to select your desired model and set the HUGGING_FACE_HUB_TOKEN.  

** For CPU support (not recommended for vLLM), build and run it in docker using this [Dockerfile](./Dockerfile).
```bash
cd <root>/swarms/playground/demos/chatbot
docker build -t llm-serving:vllm-cpu -f ~/vllm/Dockerfile.cpu .
docker run --rm --env "HF_TOKEN=<your hugging face token>" \
  --ipc=host \
  -p 8000:8000 \
  llm-serving:vllm-cpu \
  --model <Huggingface Model Path>
```

* Start the Chatbot API Server with the following shell command:

```bash
uvicorn server:app --port 8888
```

* Start the Chatbot UI

From the chatbot-ui directory:

```bash
  yarn install
  yarn run dev.  
```

## Chatbot API Server (FastAPI)

This API is written in Python and depends on FastAPI.

Follow the instructions in the [API Server README.md](./server/README.md) to install and start the API server.

## Chatbot UI (NextJS)

The chatbot-ui submodule is the frontend UI; it's a NextJS Single Page Application (SPA).

Follow the instructions in the [Chatbot-ui Submodule README.md](./chatbot-ui/README.md) to install and start the application.