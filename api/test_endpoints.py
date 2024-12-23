import time
import requests
import boto3
#from dateutil import tz


def parse_command_id(send_command_output):
    return send_command_output['Command']['CommandId']

def main():
    ec2_client = boto3.client('ec2')
    ssm_client = boto3.client('ssm')

    # Get the list of instance IDs and their states
    instances_response = ec2_client.describe_instances()

    for reservation in instances_response['Reservations']:
            for instance in reservation['Instances']:
                state = instance['State']["Name"]
                instance_id = instance['InstanceId']
                if state == 'running':

                    ip = instance["PublicIpAddress"]
                    instance_type = instance["InstanceType"]
                    BASE_URL=f"http://{ip}:80/v1"
                    target = f"{BASE_URL}/docs"
                    print(f"Starting command for instance: {instance_id} {target} {instance_type}")
                    try:
                        response = requests.get(target, timeout=8)                   
                        print(f"got response: {instance_id} {target} {instance_type} {response}")
                    except Exception as exp:
                        print(f"got error: {instance_id} {target} {instance_type} {exp}")

                    # {'AmiLaunchIndex': 0, 'ImageId': 'ami-0e2c8caa4b6378d8c',
                    #'InstanceId': 'i-0d41e4263f40babec',
                    #'InstanceType': 't3.small',
                    #'KeyName': 'mdupont-deployer-key', 'LaunchTime': datetime.datetime(2024, 12, 14, 16, 1, 50, tzinfo=tzutc()),
                    #  'Monitoring': {'State': 'disabled'},
                    #  'Placement': {'AvailabilityZone': 'us-east-1a', 'GroupName': '', 'Tenancy': 'default'}, 'PrivateDnsName': 'ip-10-0-4-18.ec2.internal', 'PrivateIpAddress': '10.0.4.18', 'ProductCodes': [],
                    #'PublicDnsName': 'ec2-3-228-14-220.compute-1.amazonaws.com',
                    #'PublicIpAddress': '3.228.14.220',
                    #  'State': {'Code': 16, 'Name': 'running'}, 'StateTransitionReason': '', 'SubnetId': 'subnet-057c90cfe7b2e5646', 'VpcId': 'vpc-04f28c9347af48b55', 'Architecture': 'x86_64',
                    #  'BlockDeviceMappings': [{'DeviceName': '/dev/sda1',
                    #                           'Ebs': {'AttachTime': datetime.datetime(2024, 12, 14, 16, 1, 50, tzinfo=tzutc()), 'DeleteOnTermination': True, 'Status': 'attached', 'VolumeId': 'vol-0257131dd2883489b'}}], 'ClientToken': 'b5864f17-9e56-2d84-fc59-811abf8e6257', 'EbsOptimized': False, 'EnaSupport': True, 'Hypervisor': 'xen', 'IamInstanceProfile':
                    #  {'Arn': 'arn:aws:iam::767503528736:instance-profile/swarms-20241213150629570500000003', 'Id': 'AIPA3FMWGOMQKC4UE2UFO'}, 'NetworkInterfaces': [
                    #      {'Association':
                    #       {'IpOwnerId': 'amazon', 'PublicDnsName': 'ec2-3-228-14-220.compute-1.amazonaws.com', 'PublicIp': '3.228.14.220'}, 'Attachment':
                    #       {'AttachTime': datetime.datetime(2024, 12, 14, 16, 1, 50, tzinfo=tzutc()), 'AttachmentId': 'eni-attach-009b54c039077324e', 'DeleteOnTermination': True, 'DeviceIndex': 0, 'Status': 'attached', 'NetworkCardIndex': 0}, 'Description': '', 'Groups': [
                    #           {'GroupName': 'swarms-20241214133959057000000001', 'GroupId': 'sg-03c9752b62d0bcfe4'}], 'Ipv6Addresses': [], 'MacAddress': '02:c9:0b:47:cb:df', 'NetworkInterfaceId': 'eni-08661c8b4777c65c7', 'OwnerId': '767503528736', 'PrivateDnsName': 'ip-10-0-4-18.ec2.internal', 'PrivateIpAddress': '10.0.4.18', 'PrivateIpAddresses': [
                    #               {'Association':
                    #                {'IpOwnerId': 'amazon', 'PublicDnsName': 'ec2-3-228-14-220.compute-1.amazonaws.com', 'PublicIp': '3.228.14.220'}, 'Primary': True, 'PrivateDnsName': 'ip-10-0-4-18.ec2.internal', 'PrivateIpAddress': '10.0.4.18'}], 'SourceDestCheck': True, 'Status': 'in-use', 'SubnetId': 'subnet-057c90cfe7b2e5646', 'VpcId': 'vpc-04f28c9347af48b55', 'InterfaceType': 'interface'}], 'RootDeviceName': '/dev/sda1', 'RootDeviceType': 'ebs', 'SecurityGroups': [
                    #                    {'GroupName': 'swarms-20241214133959057000000001', 'GroupId': 'sg-03c9752b62d0bcfe4'}], 'SourceDestCheck': True, 'Tags': [
                    #                        {'Key': 'Name', 'Value': 'swarms-size-t3.small'},
                    #                        {'Key': 'aws:ec2launchtemplate:id', 'Value': 'lt-0e618a900bd331cfe'},
                    #                        {'Key': 'aws:autoscaling:groupName', 'Value': 'swarms-size-t3.small-2024121416014474500000002f'},
                    #                        {'Key': 'aws:ec2launchtemplate:version', 'Value': '1'}], 'VirtualizationType': 'hvm', 'CpuOptions':
                    #  {'CoreCount': 1, 'ThreadsPerCore': 2}, 'CapacityReservationSpecification':
                    #  {'CapacityReservationPreference': 'open'}, 'HibernationOptions':
                    #  {'Configured': False}, 'MetadataOptions':
                    #  {'State': 'applied', 'HttpTokens': 'required', 'HttpPutResponseHopLimit': 2, 'HttpEndpoint': 'enabled', 'HttpProtocolIpv6': 'disabled', 'InstanceMetadataTags': 'disabled'}, 'EnclaveOptions':
                    #  {'Enabled': False}, 'BootMode': 'uefi-preferred', 'PlatformDetails': 'Linux/UNIX', 'UsageOperation': 'RunInstances', 'UsageOperationUpdateTime': datetime.datetime(2024, 12, 14, 16, 1, 50, tzinfo=tzutc()), 'PrivateDnsNameOptions':
                    #  {'HostnameType': 'ip-name', 'EnableResourceNameDnsARecord': False, 'EnableResourceNameDnsAAAARecord': False}, 'MaintenanceOptions':
                    #  {'AutoRecovery': 'default'}, 'CurrentInstanceBootMode': 'uefi'}

if __name__ == "__main__":
    main()
