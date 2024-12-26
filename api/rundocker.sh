#!/bin/bash

# this is the install script 
#  install_script = "/opt/swarms/api/rundocker.sh"
# called on boot.

# this is the refresh script called from ssm for a refresh
#  #refresh_script = "/opt/swarms/api/docker-boot.sh" 

# file not found
#
pwd
ls -latr
. ./.env # for secrets
set -e # stop  on any error
#export ROOT="" # empty
export WORKSOURCE="/opt/swarms/api"

adduser --disabled-password --gecos "" swarms --home "/home/swarms"  || echo ignore
git config --global --add safe.directory "/opt/swarms"
git config --global --add safe.directory "/opt/swarms-memory"

cd "/opt/swarms/" || exit 1 # "we need swarms"
git log -2 --patch | head  -1000

mkdir -p "/var/swarms/agent_workspace/"
mkdir -p "/home/swarms"


cd "/opt/swarms/" || exit 1 # "we need swarms"

mkdir -p "/var/swarms/logs"
chown -R swarms:swarms "/var/swarms/" "/home/swarms" "/opt/swarms"

#if [ -f "/var/swarms/agent_workspace/boot_fast.sh" ];
#then
#    chmod +x "/var/swarms/agent_workspace/boot_fast.sh" || echo faild
    
#    # user install but do not start
#    su -c "bash -e -x /var/swarms/agent_workspace/boot_fast.sh" swarms
#fi
cd "/opt/swarms/" || exit 1 # "we need swarms"

mkdir -p "/var/run/swarms/secrets/"
mkdir -p "/home/swarms/.cache/huggingface/hub"

set +x
OPENAI_KEY=$(aws ssm get-parameter     --name "swarms_openai_key" | jq .Parameter.Value -r )
export OPENAI_KEY
echo "OPENAI_KEY=${OPENAI_KEY}" > "/var/run/swarms/secrets/env"
set -x

## append new homedir
# check if the entry exists already before appending pls
if ! grep -q "HF_HOME" "/var/run/swarms/secrets/env"; then
       echo "HF_HOME=/home/swarms/.cache/huggingface/hub" >> "/var/run/swarms/secrets/env"
fi

if ! grep -q "^HOME" "/var/run/swarms/secrets/env"; then
    echo "HOME=/home/swarms" >> "/var/run/swarms/secrets/env"
fi

if ! grep -q "^HOME" "/var/run/swarms/secrets/env"; then
# attempt to move the workspace
    echo "WORKSPACE_DIR=\${STATE_DIRECTORY}" >> "/var/run/swarms/secrets/env"
fi

# setup the systemd service again
sed -e "s!ROOT!!g" > /etc/nginx/sites-enabled/default < "${WORKSOURCE}/nginx/site.conf"
sed -e "s!ROOT!!g" > /etc/systemd/system/swarms-docker.service < "${WORKSOURCE}/systemd/swarms-docker.service"
grep . -h -n /etc/systemd/system/swarms-docker.service

chown -R swarms:swarms /var/run/swarms/
mkdir -p /opt/swarms/api/agent_workspace/try_except_wrapper/
chown -R swarms:swarms /opt/swarms/api/


# always reload
# might be leftover on the ami,
systemctl stop swarms-uvicorn || echo ok
systemctl disable swarms-uvicorn || echo ok
rm /etc/systemd/system/swarms-uvicorn.service

systemctl daemon-reload
systemctl start swarms-docker || journalctl -xeu swarms-docker
systemctl enable swarms-docker || journalctl -xeu swarms-docker
systemctl enable nginx
systemctl start nginx

journalctl -xeu swarms-docker | tail -200 || echo oops
systemctl status swarms-docker || echo oops2

# now after swarms is up, we restart nginx
HOST="localhost"
PORT=8000
while ! nc -z $HOST $PORT; do
  sleep 1
  echo -n "."
done
echo "Port ${PORT} is now open!"

systemctl restart nginx
