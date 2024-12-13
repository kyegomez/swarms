#!/bin/bash

# EDIT: we need to make sure the instance is running
# Get the list of instance IDs
instance_ids=$(aws ec2 describe-instances --query "Reservations[*].Instances[*].InstanceId" --output text)

# Loop through each instance ID and start a session
for instance_id in $instance_ids; do
    echo "Starting session for instance: $instance_id"
    
    # Start a session and execute commands (replace with your commands)
    aws ssm start-session --target "$instance_id" --document-name "AWS-StartInteractiveCommand" --parameters 'commands=["sudo su -","tail /var/log/cloud-init-output.log"]'

done
