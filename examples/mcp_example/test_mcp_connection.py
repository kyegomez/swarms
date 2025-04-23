import requests
import time
import sys
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="DEBUG")

def test_server_connection():
    """Simple test to see if server responds at all."""
    url = "http://localhost:8000"
    
    try:
        logger.debug(f"Testing connection to {url}")
        response = requests.get(url)
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response content: {response.text[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_server_connection()