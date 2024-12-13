Starting session for instance: i-073378237c5a9dda1
+ aws ssm send-command --document-name AWS-RunShellScript --targets Key=instanceids,Values=i-073378237c5a9dda1 --parameters 'commands=["sudo su - -c \"tail /var/log/cloud-init-output.log\""]'
{
    "Command": {
        "CommandId": "820dcf47-e8d7-4c23-8e8a-bc64de2883ff",
        "DocumentName": "AWS-RunShellScript",
        "DocumentVersion": "$DEFAULT",
        "Comment": "",
        "ExpiresAfter": 1734111684.683,
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
        "RequestedDateTime": 1734104484.683,
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
}
read this and check on the results now pls


To check the results of the command you executed using AWS SSM, you can use the following command:

```bash
aws ssm list-command-invocations --command-id 820dcf47-e8d7-4c23-8e8a-bc64de2883ff --details
```

This command will provide the status and output of the command executed on your instance. Look for the `Status` and `StandardOutputContent` fields in the response to determine the result of your command.
