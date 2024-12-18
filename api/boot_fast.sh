#!/bin/bash

# to be run as swarms user
set -e
set -x
export ROOT=""
export HOME="${ROOT}/home/swarms"
unset CONDA_EXE
unset CONDA_PYTHON_EXE
export PATH="${ROOT}/var/swarms/agent_workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ls "${ROOT}/var/swarms/agent_workspace/"
. "${ROOT}/var/swarms/agent_workspace/.venv/bin/activate"

pip install  -e "${ROOT}/opt/swarms/"
cd "${ROOT}/var/swarms/"
pip install  -e "${ROOT}/opt/swarms-memory"
