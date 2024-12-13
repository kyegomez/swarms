import json

# Read JSON data from the file
with open('.cache/aws_ssm_list_commands', 'r') as file:
    data = json.load(file)

# Loop through each command
for command in data.get("Commands", []):
    command_id = command.get("CommandId")
    status = command.get("Status")

    # Check if the status is 'Success'
    if status == "Success":
        # Fetch logs of the command (you may need to implement this function)
        print(f"Fetching logs for Command ID: {command_id}")
        # Example: fetch_logs(command_id)
        # use aws ssm to fetch the logs
