#!/bin/bash
instance_ids=$(aws ec2 describe-instances --query "Reservations[*].Instances[*].InstanceId" --output text)

for instance_id in $instance_ids; do
    echo "Starting session for instance: $instance_id"
    aws ec2 terminate-instances --instance-ids "$instance_id"
done
