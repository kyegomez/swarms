#!/bin/bash
# run swarms via docker via systemd
# this script is called from ssm 
# pull the new version via systemd

# now allow for reconfigure of the systemd
export WORKSOURCE="/opt/swarms/api"
mkdir -p "/var/run/swarms/secrets/"
mkdir -p "/home/swarms/.cache/huggingface/hub"

if ! grep -q "^OPENAI_KEY" "/var/run/swarms/secrets/env"; then
    
    OPENAI_KEY=$(aws ssm get-parameter     --name "swarms_openai_key" | jq .Parameter.Value -r )
    export OPENAI_KEY
    echo "OPENAI_KEY=${OPENAI_KEY}" > "/var/run/swarms/secrets/env"
fi   

sed -e "s!ROOT!!g" > /etc/nginx/sites-enabled/default < "${WORKSOURCE}/nginx/site.conf"
sed -e "s!ROOT!!g" > /etc/systemd/system/swarms-docker.service < "${WORKSOURCE}/systemd/swarms-docker.service"
grep . -h -n /etc/systemd/system/swarms-docker.service

systemctl daemon-reload
# start and stop the service pulls the docker image
#systemctl stop swarms-docker || journalctl -xeu swarms-docker
#systemctl start swarms-docker || journalctl -xeu swarms-docker
systemctl restart swarms-docker || journalctl -xeu swarms-docker.service
systemctl enable swarms-docker || journalctl -xeu swarms-docker
