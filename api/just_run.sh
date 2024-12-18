#!/bin/bash
# review and improve
. ./.env # for secrets
set -e # stop  on any error
set -x
export BRANCH="feature/ec2"
#export ROOT="/mnt/data1/swarms"
export ROOT="" # empty
export WORKSOURCE="${ROOT}/opt/swarms/api"

adduser --disabled-password --gecos "" swarms --home "${ROOT}/home/swarms"  || echo ignore
git config --global --add safe.directory "${ROOT}/opt/swarms"
git config --global --add safe.directory "${ROOT}/opt/swarms-memory"

cd "${ROOT}/opt/swarms/" || exit 1 # "we need swarms"
git checkout --force  $BRANCH
git pull 
git log -2 --patch | head  -1000

mkdir -p "${ROOT}/var/swarms/agent_workspace/"
mkdir -p "${ROOT}/home/swarms"


cd "${ROOT}/opt/swarms/" || exit 1 # "we need swarms"
git checkout --force  $BRANCH
git pull 
git log -2 --patch | head  -1000
cp "${WORKSOURCE}/boot_fast.sh" "${ROOT}/var/swarms/agent_workspace/boot_fast.sh"
mkdir -p "${ROOT}/var/swarms/logs"
chmod +x "${ROOT}/var/swarms/agent_workspace/boot_fast.sh"
chown -R swarms:swarms "${ROOT}/var/swarms/" "${ROOT}/home/swarms" "${ROOT}/opt/swarms"

# user install but do not start
su -c "bash -e -x ${ROOT}/var/swarms/agent_workspace/boot_fast.sh" swarms

cd "${ROOT}/opt/swarms/" || exit 1 # "we need swarms"
git checkout --force  $BRANCH
git pull # $BRANCH

mkdir -p "${ROOT}/var/run/swarms/secrets/"
mkdir -p "${ROOT}/home/swarms/.cache/huggingface/hub"
# aws ssm get-parameter     --name "swarms_openai_key" > /root/openaikey.txt
export OPENAI_KEY=`aws ssm get-parameter     --name "swarms_openai_key" | jq .Parameter.Value -r `
echo "OPENAI_KEY=${OPENAI_KEY}" > "${ROOT}/var/run/swarms/secrets/env"

## append new homedir
echo "HF_HOME=${ROOT}/home/swarms/.cache/huggingface/hub" >> "${ROOT}/var/run/swarms/secrets/env"
echo "HOME=${ROOT}/home/swarms" >> "${ROOT}/var/run/swarms/secrets/env"
# attempt to move the workspace
echo 'WORKSPACE_DIR=${STATE_DIRECTORY}' >> "${ROOT}/var/run/swarms/secrets/env"

chown -R swarms:swarms ${ROOT}/var/run/swarms/
mkdir -p ${ROOT}/opt/swarms/api/agent_workspace/try_except_wrapper/
chown -R swarms:swarms ${ROOT}/opt/swarms/api/

# always reload
systemctl daemon-reload
systemctl start swarms-uvicorn || journalctl -xeu swarms-uvicorn.service
systemctl enable swarms-uvicorn || journalctl -xeu swarms-uvicorn.service
systemctl enable nginx
systemctl start nginx

journalctl -xeu swarms-uvicorn.service | tail -200 || echo oops
systemctl status swarms-uvicorn.service || echo oops2

# now after swarms is up, we restart nginx
HOST="localhost"
PORT=5474
while ! nc -z $HOST $PORT; do
  sleep 1
  echo -n "."
done
echo "Port $PORT is now open!"

systemctl restart nginx
