from swarms_client import SwarmsClient
from dotenv import load_dotenv
import os

load_dotenv()

client = SwarmsClient(api_key=os.getenv("SWARMS_API_KEY"))

response = client.client.rate.get_limits()
print(response)

print(client.health.check())
