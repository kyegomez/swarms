import time

import boto3
#from dateutil import tz


def parse_command_id(send_command_output):
    return send_command_output['Command']['CommandId']

def main():
    ec2_client = boto3.client('ec2')
    ssm_client = boto3.client('ssm')

    # Get the list of instance IDs and their states
    instances_response = ec2_client.describe_instances()
    
    instances = [
        (instance['InstanceId'], instance['State']['Name'])
        for reservation in instances_response['Reservations']
        for instance in reservation['Instances']
    ]
    for reservation in instances_response['Reservations']:
        for instance in reservation['Instances']:
            print(instance)
            instance_id = instance['InstanceId']
            state = instance['State']['Name']            
            if state == 'running':
                print(f"Starting command for instance: {instance_id}")    
                print(f"aws ssm start-session --target  {instance_id}")
        
if __name__ == "__main__":
    main()
