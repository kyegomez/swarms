`sudo bash ./install.sh`

to redo all the steps remove the lock files

`rm ${ROOT}/opt/swarms/install/* `

or in my system:
```
export ROOT=/mnt/data1/swarms
sudo rm ${ROOT}/opt/swarms/install/*
```

rerun
```
export ROOT=/mnt/data1/swarms; 
sudo rm ${ROOT}/opt/swarms/install/*; 
sudo bash ./install.sh
```
* setup
To install on linux: 
https://docs.aws.amazon.com/systems-manager/

```
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "session-manager-plugin.deb"
sudo dpkg  -i ./session-manager-plugin.deb 
```

* run

To redo the installation steps for the Swarms tool on your system, follow these commands sequentially:

1. Set the ROOT variable:
   ```bash
   export ROOT=/mnt/data1/swarms
   ```

2. Remove the lock files:
   ```bash
   sudo rm ${ROOT}/opt/swarms/install/*
   ```

3. Run the installation script again:
   ```bash
   sudo bash ./install.sh
   ```

For setting up the Session Manager plugin on Linux, you can follow these commands:

1. Download the Session Manager plugin:
   ```bash
   curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "session-manager-plugin.deb"
   ```

2. Install the plugin:
   ```bash
   sudo dpkg -i ./session-manager-plugin.deb
   ```

After that, you can run your desired commands or workflows.** get the instance id
`aws ec2 describe-instances`

** start a session
`aws ssm start-session --target  i-XXXX`

** on the machine:
```
sudo su - 
tail /var/log/cloud-init-output.log 
```

Convert this to an automation of your choice to run all the steps
and run this on all the instances

To get the instance ID and start a session using AWS CLI, follow these steps:

1. **Get the Instance ID:**
   Run the following command to list your instances and their details:
   ```bash
   aws ec2 describe-instances
   ```

2. **Start a Session:**
   Replace `i-XXXX` with your actual instance ID from the previous step:
   ```bash
   aws ssm start-session --target i-XXXX
   ```

3. **On the Machine:**
   After starting the session, you can execute the following commands:
   ```bash
   sudo su -
   tail /var/log/cloud-init-output.log
   ```


