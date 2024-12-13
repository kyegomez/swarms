#!/bin/python3
# rewrite this to aslo cancel-spot-instance-requests

import boto3

# Create an EC2 client
ec2_client = boto3.client('ec2')

# Retrieve instance IDs
response = ec2_client.describe_instances()


for reservation in response['Reservations'] :
  for instance in reservation['Instances']:
        print( instance)

instance_ids = [instance['InstanceId']
                 for reservation in response['Reservations']
                 for instance in reservation['Instances']]

# Terminate instances
for instance_id in instance_ids:
    print(f"Terminating instance: {instance_id}")
    ec2_client.terminate_instances(InstanceIds=[instance_id])

# Check the status of the terminated instances
terminated_instances = ec2_client.describe_instances(InstanceIds=instance_ids)
for reservation in terminated_instances['Reservations']:
    for instance in reservation['Instances']:
        print(f"Instance ID: {instance['InstanceId']}, State: {instance['State']['Name']}")
