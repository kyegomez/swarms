# RAG Chatbot Server

* This server is currently used to host a conversational RAG (Retrieval Augmented Generation) Chatbot.  

* It is currently set up to use OpenAI or vLLM (or other OpenAI compatible APIs such as available via LMStudio or Ollama).  

* Support for Metal is also available if configured in the .env file via the USE_METAL environment variable, but this doesn't apply when using vLLM.  

* Switching from vLLM to another host like Olama requires commenting/uncommenting some code at this time, but will be dynamic later.  

* Locally hosted or cloud (vector storage)[./vector_storage.py]

## Prerequisites

* Install (Redis Stack)[https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/]

* (Firecrawl)[https://docs.firecrawl.dev/contributing]

## Start Redis Stack and Firecrawl

```bash
    sudo apt-get install redis-stack-server
    sudo systemctl enable redis-stack-server
    sudo systemctl start redis-stack-server
    cd ~/firecrawl/apps/api
    pnpm run workers
    pnpm run start
```

## Running vLLM

Running vLLM in a docker container saves a lot of trouble.  Use dockerRunVllm.sh to set up and start vLLM.  This command will allow you to control vLLM using docker commands:

```bash
docker stop vllm

docker start vllm

docker attach vllm
```

Run the dockerRunVllm.sh command again to get a fresh copy of the latest vLLM docker image (you will be prompted to rename or remove the existing one if the name is the same.)

## Starting the Chatbot API Server

In order to start the server you have to run uvicorn or FastAPI CLI or use the following launch.json in VSCode/Cursor or whatever to debug it.

### Start server with uvicorn

Run the following shell cmd:

```bash
uvicorn server:app --port 8888
```

To debug using uvicorn use this launch.json configuration:

```json
"configurations": [
        {
            "name": "Python: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "swarms.server.server:app",  // Use dot notation for module path
                "--reload",
                "--port",
                "8888"
            ],
            "jinja": true,
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/swarms"
            }
        }
    ]

```

### Start server using FastAPI CLI

You can run the Chatbot server in production mode using FastAPI CLI:

```bash
fastapi run swarms/server/server.py --port 8888
```

To run in dev mode use this command:

```bash
fastapi dev swarms/server/server.py --port 8888
```
