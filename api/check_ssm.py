import os
import json
import boto3

# Create .cache directory if it doesn't exist
os.makedirs('.cache', exist_ok=True)

def cache(name, value):
    cache_file = f'.cache/{name}'
    if not os.path.isfile(cache_file):
        with open(cache_file, 'w') as f:
            f.write(value)

# Initialize Boto3 SSM client
ssm = boto3.client('ssm')

# List commands from AWS SSM
response = ssm.list_commands()

cache("aws_ssm_list_commands", response)

# Retrieve commands
print(response)
commands = response["Commands"]
run_ids = [cmd['CommandId'] for cmd in commands]
print(f"RUNIDS: {run_ids}")

# Check the status of each command
for command in commands:
    #print(command)
    command_id = command['CommandId']
    status = command['Status']
    #eG: command= {'CommandId': '820dcf47-e8d7-4c23-8e8a-bc64de2883ff', 'DocumentName': 'AWS-RunShellScript', 'DocumentVersion': '$DEFAULT', 'Comment': '', 'ExpiresAfter': datetime.datetime(2024, 12, 13, 12, 41, 24, 683000, tzinfo=tzlocal()), 'Parameters': {'commands': ['sudo su - -c "tail /var/log/cloud-init-output.log"']}, 'InstanceIds': [], 'Targets': [{'Key': 'instanceids', 'Values': ['i-073378237c5a9dda1']}], 'RequestedDateTime': datetime.datetime(2024, 12, 13, 10, 41, 24, 683000, tzinfo=tzlocal()), 'Status': 'Success', 'StatusDetails': 'Success', 'OutputS3Region': 'us-east-1', 'OutputS3BucketName': '', 'OutputS3KeyPrefix': '', 'MaxConcurrency': '50', 'MaxErrors': '0', 'TargetCount': 1, 'CompletedCount': 1, 'ErrorCount': 0, 'DeliveryTimedOutCount': 0, 'ServiceRole': '', 'NotificationConfig': {'NotificationArn': '', 'NotificationEvents': [], 'NotificationType': ''}, 'CloudWatchOutputConfig': {'CloudWatchLogGroupName': '', 'CloudWatchOutputEnabled': False}, 'TimeoutSeconds': 3600, 'AlarmConfiguration': {'IgnorePollAlarmFailure': False, 'Alarms': []}, 'TriggeredAlarms': []}], 'ResponseMetadata': {'RequestId': '535839c4-9b87-4526-9c01-ed57f07d21ef', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'Server', 'date': 'Fri, 13 Dec 2024 16:58:53 GMT', 'content-type': 'application/x-amz-json-1.1', 'content-length': '2068', 'connection': 'keep-alive', 'x-amzn-requestid': '535839c4-9b87-4526-9c01-ed57f07d21ef'}, 'RetryAttempts': 0}}
    
    if status == "Success":
        print(f"Check logs of {command_id}")
        # use ssm to  fetch logs using CommandId

        # Assuming you have the command_id from the previous command output
        command_id = command['CommandId']
        instance_id = command['Targets'][0]['Values'][0]  # Get the instance ID

        # Fetching logs using CommandId
        log_response = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )
        print(log_response['StandardOutputContent'])  # Output logs
        print(log_response['StandardErrorContent'])    # Error logs (if any)
        print(f"aws ssm start-session --target  {instance_id}")


