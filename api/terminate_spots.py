#!/bin/python3

import boto3

# Create an EC2 client
ec2_client = boto3.client('ec2')

# Retrieve instance IDs and Spot Instance Request IDs
response = ec2_client.describe_instances()
instance_ids = []
spot_request_ids = []

for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(instance)
        instance_ids.append(instance['InstanceId'])
        if 'SpotInstanceRequestId' in instance:
            spot_request_ids.append(instance['SpotInstanceRequestId'])

# Terminate instances
for instance_id in instance_ids:
    print(f"Terminating instance: {instance_id}")
    ec2_client.terminate_instances(InstanceIds=[instance_id])

# Cancel Spot Instance Requests
for spot_request_id in spot_request_ids:
    print(f"Cancelling Spot Instance Request: {spot_request_id}")
    ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])

# Check the status of the terminated instances
terminated_instances = ec2_client.describe_instances(InstanceIds=instance_ids)
for reservation in terminated_instances['Reservations']:
    for instance in reservation['Instances']:
        print(f"Instance ID: {instance['InstanceId']}, State: {instance['State']['Name']}")
