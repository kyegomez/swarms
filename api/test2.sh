
#/mnt/data1/swarms/var/run/uvicorn/env/bin/uvicorn 
#        --no-access-log \

#python -m pdb
#/mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/uvicorn \

. /mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/activate
pip install hunter
/mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/python3 ~mdupont/2024/05/swarms/api/uvicorn_runner.py
