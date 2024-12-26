# review
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/swarms/
RUN apt update
RUN apt install -y git
RUN apt install --allow-change-held-packages -y python3-virtualenv
RUN apt install --allow-change-held-packages -y expect
RUN apt install --allow-change-held-packages -y jq netcat-traditional # missing packages

# Install Python dependencies

RUN mkdir -p /var/swarms/agent_workspace/
RUN adduser --disabled-password --gecos "" swarms --home "/home/swarms"
RUN chown  -R swarms:swarms /var/swarms/agent_workspace
USER swarms
RUN python3 -m venv /var/swarms/agent_workspace/.venv/

# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cublas-cu12==12.4.5.8
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cuda-cupti-cu12==12.4.127
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cuda-nvrtc-cu12==12.4.127
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cuda-runtime-cu12==12.4.127
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cudnn-cu12==9.1.0.70
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cufft-cu12==11.2.1.3
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-curand-cu12==10.3.5.147
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cusolver-cu12==11.6.1.9
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-cusparse-cu12==12.3.1.170
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-nccl-cu12==2.21.5
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-nvjitlink-cu12==12.4.127
# RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install nvidia-nvtx-cu12==12.4.127

RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install --upgrade pip

# 15l
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install  aiofiles==24.1.0 \
    aiohappyeyeballs==2.4.4 \
    aiosignal==1.3.2 \
    frozenlist==1.5.0 \
    aiohttp==3.11.11 \
    attrs==24.3.0 \
    annotated-types==0.7.0 \
    anyio==4.7.0 \
    sniffio==1.3.1 \
    typing_extensions==4.12.2 \
    asyncio==3.4.3 \
    multidict==6.1.0 \
    propcache==0.2.1 \
    yarl==1.18.3 \
    idna==3.10

RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install certifi==2024.12.14 \
    chardet==5.2.0 \
    charset-normalizer==3.4.0 \
     click==8.1.8  \
     dataclasses-json==0.6.7 \
    marshmallow==3.23.2 \
    typing-inspect==0.9.0 \
    packaging==23.2 \
    mypy-extensions==1.0.0 \
    dill==0.3.9 \
    distro==1.9.0 \
    docstring_parser==0.16 \
    filelock==3.16.1 \
    fastapi==0.115.6 \
    starlette==0.41.3 \
    pydantic==2.10.4 \
    pydantic_core==2.27.2
# 15s


RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install GPUtil==1.4.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install Jinja2==3.1.5 MarkupSafe==3.0.2 PyYAML==6.0.2 Pygments==2.18.0  SQLAlchemy==2.0.36 fsspec==2024.12.0 greenlet==3.1.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install h11==0.14.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install httpcore==1.0.7
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install httpx==0.27.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install huggingface-hub==0.27.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install importlib_metadata==8.5.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install iniconfig==2.0.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install jiter==0.8.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install jsonpatch==1.33
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install jsonpointer==3.0.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install jsonschema-specifications==2024.10.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install jsonschema==4.23.0
# Huge 
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install langchain-community==0.0.29   \
    langchain-core==0.1.53 \
    langsmith==0.1.147 \
    numpy==1.26.4 \
    orjson==3.10.12 \
    requests-toolbelt==1.0.0 \
    tenacity==8.5.0


RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install loguru==0.7.3
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install lxml==5.3.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install markdown-it-py==3.0.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install mdurl==0.1.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install mpmath==1.3.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install msgpack==1.1.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install multiprocess==0.70.17
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install mypy-protobuf==3.6.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install networkx==3.4.2

RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install ollama==0.4.4
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install openai==1.58.1

RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pathos==0.3.3
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pathspec==0.12.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install platformdirs==4.3.6
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pluggy==1.5.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pox==0.3.5
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install ppft==1.7.6.9
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install protobuf==5.29.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install psutil==6.1.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pytesseract==0.3.13
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pytest==8.3.4
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install python-dateutil==2.9.0.post0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install python-docx==1.1.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install python-dotenv==1.0.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install python-magic==0.4.27
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pytz==2024.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install ratelimit==2.2.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install referencing==0.35.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install regex==2024.11.6
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install reportlab==4.2.5

RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install requests==2.32.3
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install rich==13.9.4
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install rpds-py==0.22.3
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install safetensors==0.4.5
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install sentry-sdk==2.19.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install six==1.17.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install sympy==1.13.1 #  10 sec
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install termcolor==2.5.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install tiktoken==0.8.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install tokenizers==0.21.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install toml==0.10.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install tqdm==4.67.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install types-chardet==5.0.4.6
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install types-protobuf==5.29.1.20241207
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install types-pytz==2024.2.0.20241221
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install types-toml==0.10.8.20240310
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install tzdata==2024.2
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install urllib3==2.3.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install uvicorn==0.34.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install zipp==3.21.0


## dev toools
#RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install ruff==0.4.4
#RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install black==24.10.0

## SLOW
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install doc-master==0.0.2
# needs qt
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install clusterops==0.1.6

RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pandas==2.2.3  # 13s
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install litellm==1.55.9 # 11s
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install transformers==4.47.1 # 12s
#RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install triton==3.1.0 #  18s
# Big ones
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pillow==11.0.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install pypdf==5.1.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install ray==2.40.0
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install torch==2.5.1
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install together==1.3.10

###
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install git+https://github.com/jmikedupont2/swarm-models@main#egg=swarm-models
# swarm-models==0.2.7 #  BIG 55 sec
#RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install -r /opt/swarms/requirements.txt
RUN git config --global --add safe.directory "/opt/swarms"
#RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install uvicorn fastapi

COPY swarms /opt/swarms/swarms
COPY pyproject.toml /opt/swarms/
COPY README.md /opt/swarms/
RUN /var/swarms/agent_workspace/.venv/bin/python -m pip install -e /opt/swarms/
#COPY requirements.txt .
# things that change
COPY api/main.py /opt/swarms/api/main.py
WORKDIR /var/swarms/agent_workspace
CMD ["/usr/bin/unbuffer", "/var/swarms/agent_workspace/.venv/bin/uvicorn", "--proxy-headers", "--forwarded-allow-ips='*'", "--workers=4", "--port=8000",    "--reload-delay=30",  "main:create_app"]
