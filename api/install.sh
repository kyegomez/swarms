#!/bin/bash
# review and improve
. ./.env # for secrets
set -e # stop  on any error
set -x

export ROOT="/mnt/data1/swarms"
export WORKSOURCE="${ROOT}/opt/swarms/api"

if [ ! -d "${ROOT}/opt/swarms/install/" ]; then
    mkdir -p "${ROOT}/opt/swarms/install"
fi

if [ ! -f "${ROOT}/opt/swarms/install/apt.txt" ]; then
    sudo apt update
    sudo apt install --allow-change-held-packages -y git virtualenv
    echo 1 >"${ROOT}/opt/swarms/install/apt.txt"
fi

if [ ! -f "${ROOT}/opt/swarms/install/setup.txt" ]; then
    #rm -rf ./src/swarms # oops
    adduser swarms --home "${ROOT}/home/swarms" || echo ignore
    git config --global --add safe.directory "${ROOT}/opt/swarms"
    git config --global --add safe.directory "${ROOT}/opt/swarms-memory"
    # we should have done this
    if [ ! -d "${ROOT}/opt/swarms/" ];
    then
	git clone https://github.com/jmikedupont2/swarms "${ROOT}/opt/swarms/"
    fi    
    pushd "${ROOT}/opt/swarms/" || exit 1 # "we need swarms"
    git remote add local /time/2024/05/swarms/ || git remote set-url local /time/2024/05/swarms/ 
    git fetch local 
    git checkout feature/ec2 # switch branches
    git pull local feature/ec2
    popd || exit 2    
    if [ ! -d "${ROOT}/opt/swarms-memory/" ];
    then
	git clone https://github.com/The-Swarm-Corporation/swarms-memory "${ROOT}/opt/swarms-memory"
    fi    
    # where the swarms will run
    mkdir -p "${ROOT}/var/swarms/agent_workspace/"
    mkdir -p "${ROOT}/home/swarms"
    chown -R swarms:swarms "${ROOT}/var/swarms/agent_workspace" "${ROOT}/home/swarms"    

    # now for my local setup I aslo need to do this or we have to change the systemctl home var
    #mkdir -p "/home/swarms"
    #chown -R swarms:swarms "/home/swarms"    

    # copy the run file from git
    cp "${WORKSOURCE}/boot.sh" "${ROOT}/var/swarms/agent_workspace/boot.sh"
    mkdir -p "${ROOT}/var/swarms/logs"
    chmod +x "${ROOT}/var/swarms/agent_workspace/boot.sh"
    chown -R swarms:swarms "${ROOT}/var/swarms/" "${ROOT}/home/swarms" "${ROOT}/opt/swarms"

    echo 1 >"${ROOT}/opt/swarms/install/setup.txt"
fi

if [ ! -f "${ROOT}/opt/swarms/install/boot.txt" ]; then
    # user install but do not start
    su -c "bash -e -x ${ROOT}/var/swarms/agent_workspace/boot.sh" swarms
    echo 1 >"${ROOT}/opt/swarms/install/boot.txt"
fi
    

## pull

if [ ! -f "${ROOT}/opt/swarms/install/pull.txt" ]; then
    pushd "${ROOT}/opt/swarms/" || exit 1 # "we need swarms"
    git fetch local 
    git checkout feature/ec2 # switch branches
    git pull local feature/ec2
    popd || exit 2    
    #echo 1 >"${ROOT}/opt/swarms/install/pull.txt"
fi


if [ ! -f "${ROOT}/opt/swarms/install/config.txt" ]; then
    #WorkingDirectory=ROOT/var/run/swarms/
    # create the secrets
    mkdir -p "${ROOT}/var/run/swarms/secrets/"
    echo "OPENAI_KEY=${OPENAI_KEY}" > "${ROOT}/var/run/swarms/secrets/env"

    # cache 
    mkdir -p "${ROOT}/home/swarms/.cache/huggingface/hub"
    
    ## append new homedir
    echo "TRANSFORMERS_CACHE=${ROOT}/home/swarms/.cache/huggingface/hub" >> "${ROOT}/var/run/swarms/secrets/env"
    echo "HOME=${ROOT}/home/swarms" >> "${ROOT}/var/run/swarms/secrets/env"
    #EnvironmentFile=ROOT/var/run/swarms/secrets/env
    #ExecStart=ROOT/var/run/uvicorn/env/bin/uvicorn \
	#	--uds ROOT/run/uvicorn/uvicorn-swarms-api.sock \
    echo 1 >"${ROOT}/opt/swarms/install/config.txt"    
fi
	
# if [ ! -f "${ROOT}/opt/swarms/install/uvicorn.txt" ]; then    
# # we create a second installation of unicorn so agents cannot mess it up.
#     mkdir -p "${ROOT}/var/run/uvicorn/env/"
#     if [ ! -f "${ROOT}/var/run/uvicorn/env/" ];
#     then
# 	virtualenv "${ROOT}/var/run/uvicorn/env/"
#     fi
#     . "${ROOT}/var/run/uvicorn/env/bin/activate"
#     pip install  uvicorn   
#     echo 1 >"${ROOT}/opt/swarms/install/uvicorn.txt"
# fi

if [ ! -f "${ROOT}/opt/swarms/install/nginx.txt" ]; then
    mkdir -p ${ROOT}/var/log/nginx/swarms/
fi
# reconfigure
# now we setup the service and  replace root in the files

#echo  cat "${WORKSOURCE}/nginx/site.conf" \| sed -e "s!ROOT!${ROOT}!g" 
cat "${WORKSOURCE}/nginx/site.conf"| sed -e "s!ROOT!${ROOT}!g" > /etc/nginx/sites-enabled/default
#cat /etc/nginx/sites-enabled/default

# create sock
mkdir -p ${ROOT}/run/uvicorn/
chown -R swarms:swarms ${ROOT}/run/uvicorn

# ROOT/var/run/swarms/uvicorn-swarms-api.sock;
#    access_log ROOT/var/log/nginx/swarms/access.log;
#    error_log ROOT/var/log/nginx/swarms/error.log;
#echo cat "${WORKSOURCE}/systemd/uvicorn.service" \| sed -e "s!ROOT!/${ROOT}/!g"
cat "${WORKSOURCE}/systemd/uvicorn.service" | sed -e "s!ROOT!${ROOT}!g" > /etc/systemd/system/swarms-uvicorn.service


# if [ -f ${ROOT}/etc/systemd/system/swarms-uvicorn.service ];
# then
#     cp ${ROOT}/etc/systemd/system/swarms-uvicorn.service /etc/systemd/system/swarms-uvicorn.service
# else
#     # allow for editing as non root
#     mkdir -p ${ROOT}/etc/systemd/system/
#     cp /etc/systemd/system/swarms-uvicorn.service ${ROOT}/etc/systemd/system/swarms-uvicorn.service      
# fi

# 
#chown -R mdupont:mdupont ${ROOT}/etc/systemd/system/
#/run/uvicorn/
# triage
chown -R swarms:swarms ${ROOT}/var/run/swarms/


#cat /etc/systemd/system/swarms-uvicorn.service
# always reload
systemctl daemon-reload
#    systemctl start swarms-uvicorn || systemctl status swarms-uvicorn.service  && journalctl -xeu swarms-uvicorn.service
systemctl start swarms-uvicorn || journalctl -xeu swarms-uvicorn.service
# systemctl status swarms-uvicorn.service
# journalctl -xeu swarms-uvicorn.service
systemctl enable swarms-uvicorn || journalctl -xeu swarms-uvicorn.service
service nginx restart
