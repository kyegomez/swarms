
#/mnt/data1/swarms/var/run/uvicorn/env/bin/uvicorn 
#        --no-access-log \

#python -m pdb
#/mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/uvicorn \

. /mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/activate
/mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/python3 ~mdupont/2024/05/swarms/api/uvicorn_runner.py \
    --proxy-headers \
    --port=54748 \
    --forwarded-allow-ips='*' \
    --workers=1 \
    --log-level=debug \
    --uds /mnt/data1/swarms/run/uvicorn/uvicorn-swarms-api.sock \
    main:app

#        _.asgi:application
