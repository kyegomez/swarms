
/usr/bin/unbuffer /var/swarms/agent_workspace/.venv/bin/uvicorn --proxy-headers /opt/swarms/api/main.py:create_app
