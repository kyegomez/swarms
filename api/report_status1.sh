
send_command_output='{
	    "Command": {
        "CommandId": "c019b2a0-0884-45df-af55-da8bb3d18dc5",
        "DocumentName": "AWS-RunShellScript",
        "DocumentVersion": "$DEFAULT",
        "Comment": "",
        "ExpiresAfter": 1734112021.438,
        "Parameters": {
            "commands": [
                "sudo su - -c \"tail /var/log/cloud-init-output.log\""
            ]
        },
        "InstanceIds": [],
        "Targets": [
            {
                "Key": "instanceids",
                "Values": [
                    "i-073378237c5a9dda1"
                ]
            }
        ],
        "RequestedDateTime": 1734104821.438,
        "Status": "Pending",
        "StatusDetails": "Pending",
        "OutputS3Region": "us-east-1",
        "OutputS3BucketName": "",
        "OutputS3KeyPrefix": "",
        "MaxConcurrency": "50",
        "MaxErrors": "0",
        "TargetCount": 0,
        "CompletedCount": 0,
        "ErrorCount": 0,
        "DeliveryTimedOutCount": 0,
        "ServiceRole": "",
        "NotificationConfig": {
            "NotificationArn": "",
            "NotificationEvents": [],
            "NotificationType": ""
        },
        "CloudWatchOutputConfig": {
            "CloudWatchLogGroupName": "",
            "CloudWatchOutputEnabled": false
        },
        "TimeoutSeconds": 3600,
        "AlarmConfiguration": {
            "IgnorePollAlarmFailure": false,
            "Alarms": []
        },
        "TriggeredAlarms": []
    }
}'

#++ parse_command_id send_command_output
#++ local send_command_output=send_command_output

echo $send_command_output | jq -r '.Command.CommandId'
