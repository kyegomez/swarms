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

    for instance_id, state in instances:
        if state == 'running':
            print(f"Starting command for instance: {instance_id}")

            # Send command to the instance
            send_command_output = ssm_client.send_command(
                DocumentName="AWS-RunShellScript",
                Targets=[{"Key": "instanceids", "Values": [instance_id]}],
                Parameters={'commands': ['sudo su - -c "tail /var/log/cloud-init-output.log"']}
            )

            # Get the command ID
            command_id = parse_command_id(send_command_output)

            # Check the command status every second for 4 seconds
            for _ in range(4):
                time.sleep(20)
                command_status = ssm_client.list_command_invocations(CommandId=command_id, Details=True)
                
                print(command_status)
                for invocation in command_status['CommandInvocations']:
                    if invocation['Status'] == 'Success':
                        for plugin in invocation['CommandPlugins']:
                            if plugin['Status'] == 'Success':
                                print(f"Output from instance {instance_id}:\n{plugin['Output']}")
                            else:
                                print(f"Error in plugin execution for instance {instance_id}: {plugin['StatusDetails']}")
                    else:
                        print(f"Command for instance {instance_id} is still in progress... Status: {invocation['Status']}")


if __name__ == "__main__":
    main()
