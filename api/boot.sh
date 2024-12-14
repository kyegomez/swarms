#!/bin/bash

# to be run as swarms user
set -e
set -x
export ROOT=""
export HOME="${ROOT}/home/swarms"
unset CONDA_EXE
unset CONDA_PYTHON_EXE
export PATH="${ROOT}/var/swarms/agent_workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [ ! -f "${ROOT}/var/swarms/agent_workspace/.venv/" ];
then
   virtualenv "${ROOT}/var/swarms/agent_workspace/.venv/"
fi
ls "${ROOT}/var/swarms/agent_workspace/"
. "${ROOT}/var/swarms/agent_workspace/.venv/bin/activate"

pip install fastapi   uvicorn  termcolor
# these are tried to be installed by the app on boot
pip install sniffio pydantic-core httpcore exceptiongroup annotated-types pydantic anyio httpx ollama
pip install  -e "${ROOT}/opt/swarms/"
cd "${ROOT}/var/swarms/"
pip install  -e "${ROOT}/opt/swarms-memory"
pip install "fastapi[standard]"
pip install "loguru"
pip install "hunter" # for tracing
pip install  pydantic==2.8.2
pip install pathos || echo oops
pip freeze
# launch as systemd
# python /opt/swarms/api/main.py
