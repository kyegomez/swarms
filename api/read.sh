# #read this file one command at a time and loop over it in python
# # read out the attributes of each object into variables in the loop
# # tail .cache/aws_ssm_list_commands
# # {"CommandId":"820dcf47-e8d7-4c23-8e8a-bc64de2883ff","DocumentName":"AWS-RunShellScript","DocumentVersion":"$DEFAULT","Comment":"","ExpiresAfter":1734111684.683,"Parameters":{"commands":["sudo su - -c \"tail /var/log/cloud-init-output.log\""]},"InstanceIds":[],"Targets":[{"Key":"instanceids","Values":["i-073378237c5a9dda1"]}],"RequestedDateTime":1734104484.683,"Status":"Success","StatusDetails":"Success","OutputS3Region":"us-east-1","OutputS3BucketName":"","OutputS3KeyPrefix":"","MaxConcurrency":"50","MaxErrors":"0","TargetCount":1,"CompletedCount":1,"ErrorCount":0,"DeliveryTimedOutCount":0,"ServiceRole":"","NotificationConfig":{"NotificationArn":"","NotificationEvents":[],"NotificationType":""},"CloudWatchOutputConfig":{"CloudWatchLogGroupName":"","CloudWatchOutputEnabled":false},"TimeoutSeconds":3600,"AlarmConfiguration":{"IgnorePollAlarmFailure":false,"Alarms":[]},"TriggeredAlarms":[]}

# # read this result
# commands=$(jq -e ".Commands[]" -c .cache/aws_ssm_list_commands )
# for each command {
#     id=getid
#     status=get status
#     if status == successs:
#        fetch logs of commnd id
# }

rewrite in python

Here's a Python script that reads the JSON data from the specified file, extracts the attributes of each command, and processes them in a loop:

```python
```

Make sure to implement the `fetch_logs(command_id)` function according to your requirements.
