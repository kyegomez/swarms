import os
import json
from dotenv import load_dotenv
from swarms_client import SwarmsClient

load_dotenv()

client = SwarmsClient(api_key=os.getenv("SWARMS_API_KEY"))

print(json.dumps(client.models.list_available(), indent=4))
print(json.dumps(client.health.check(), indent=4))
print(json.dumps(client.swarms.get_logs(), indent=4))
print(json.dumps(client.client.rate.get_limits(), indent=4))
print(json.dumps(client.swarms.check_available(), indent=4))
