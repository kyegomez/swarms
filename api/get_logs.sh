rewrite in python boto3
mkdir -p .cache

cache(){
    name=".cache/${1}"
    value="${2}"
    if [ ! -f $name ];
    then
	echo $value > $name
    fi
}

aws_ssm_list_commands=$(aws ssm list-commands)
cache "aws_ssm_list_commands" "$aws_ssm_list_commands"
#command_status=$(echo "${aws_ssm_list_commands}" | jq  '.Commands[]|[.CommandId,.Status]')
commands=$(echo "${aws_ssm_list_commands}" | jq  '.Commands[]')
#run_ids=$(echo $command_status | jq  .[0] -r)
run_ids=$(echo $command_status | jq -r)
echo RUNIDS:$run_ids;

while read -r instance_id state; do
    if [[ $state == "Success" ]]; then
	echo "Check $instance_id"
	#             "DocumentName": "AWS-RunShellScript",
#             "Parameters": {
#                 "commands": [
#                     "sudo su - -c \"tail /var/log/cloud-init-output.log\""
#             "Status": "Success",
#             "CompletedCount": 1,
#             "ErrorCount": 0,
#             "DeliveryTimedOutCount": 0,

    fi
done  <<< "$run_ids"



It seems you might be looking for a complete Python script that replicates the functionality of your provided shell script using Boto3. Here’s an example:

```python
```

This Python script uses Boto3 to list SSM commands, caches them in a file, and checks the status of each command. Adjust the logic as needed to fit your specific processing requirements.
