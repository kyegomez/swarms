
# # Get the list of instance IDs and their states
# instances=$(aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId,State.Name]" --output text)

# # aws ssm send-command --document-name AWS-RunShellScript --targets Key=instanceids,Values=i-073378237c5a9dda1 --parameters 'commands=["sudo su - -c \"tail /var/log/cloud-init-output.log\""]'

# parse_command_id(){
#     # send_command_output
#     local send_command_output=$1
#     echo "$send_command_output" | jq -r '.Command.CommandId'
# }

# # Loop through each instance ID and state
# while read -r instance_id state; do
#     if [[ $state == "running" ]]; then
#         echo "Starting session for instance: $instance_id"
        
#         # Start a session and execute commands (replace with your commands)
#         #aws ssm start-session --target "$instance_id" --document-name "AWS-StartInteractiveCommand" --parameters 'commands=["sudo su -","tail /var/log/cloud-init-output.log"]'

# 	#--target "$instance_id"
# 	send_command_output=$(aws ssm send-command --document-name "AWS-RunShellScript" --targets "Key=instanceids,Values=$instance_id" --parameters 'commands=["sudo su - -c \"tail /var/log/cloud-init-output.log\""]')

	
# 	# now get the command id
# 	command_id=$(parse_command_id send_command_output)
	
# 	# now for 4 times, sleep 1 sec,
# 	for i in {1..4}; do
# 	    sleep 1
# 	    command_status=$(aws ssm list-command-invocations --command-id "$command_id" --details)
# 	    echo "$command_status"
# 	done

#     fi
# done <<< "$instances"
# # rewrite in python


# Here is the equivalent Python script using `boto3` to interact with AWS SSM:

# ```python
import boto3
import time

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
                time.sleep(1)
                command_status = ssm_client.list_command_invocations(CommandId=command_id, Details=True)
                print(command_status)

if __name__ == "__main__":
    main()
