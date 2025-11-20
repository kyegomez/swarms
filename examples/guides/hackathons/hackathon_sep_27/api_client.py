import os
from dotenv import load_dotenv
from swarms_client import SwarmsClient

load_dotenv()

client = SwarmsClient(api_key=os.getenv("SWARMS_API_KEY"))

print(client.health.check())
print(client.client.rate.get_limits())
