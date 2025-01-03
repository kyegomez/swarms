"""
Simple test script for SkyServe API using requests.
No test framework dependencies - just pure requests and assertions.
"""

import time
import requests
from typing import Any

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


def assert_equals(actual: Any, expected: Any, message: str = ""):
    """Simple assertion helper."""
    if actual != expected:
        raise AssertionError(
            f"{message}\nExpected: {expected}\nGot: {actual}"
        )


def test_create_service() -> str:
    """Test service creation and return the service name."""
    print("\n🧪 Testing service creation...")

    payload = {
        "code": """
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
        """,
        "requirements": ["fastapi", "uvicorn"],
        "name": "test_service",
        "num_cpus": 2,
        "memory": 4,
    }

    response = requests.post(
        f"{BASE_URL}/services/", json=payload, headers=HEADERS
    )

    assert_equals(
        response.status_code, 201, "Service creation failed"
    )
    data = response.json()
    assert "service_name" in data, "Response missing service_name"
    assert "endpoint" in data, "Response missing endpoint"

    print("✅ Service created successfully!")
    return data["service_name"]


def test_list_services(expected_service_name: str):
    """Test listing services."""
    print("\n🧪 Testing service listing...")

    response = requests.get(f"{BASE_URL}/services/")
    assert_equals(response.status_code, 200, "Service listing failed")

    services = response.json()
    assert isinstance(services, list), "Expected list of services"

    # Find our service in the list
    service_found = False
    for service in services:
        if service["name"] == expected_service_name:
            service_found = True
            break

    assert (
        service_found
    ), f"Created service {expected_service_name} not found in list"
    print("✅ Services listed successfully!")


def test_update_service(service_name: str):
    """Test service update."""
    print("\n🧪 Testing service update...")

    update_payload = {
        "code": """
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Updated World"}
        """,
        "requirements": ["fastapi", "uvicorn"],
        "name": service_name,
        "num_cpus": 2,
        "memory": 4,
    }

    response = requests.put(
        f"{BASE_URL}/services/{service_name}",
        json=update_payload,
        headers=HEADERS,
        params={"mode": "gradual"},
    )

    assert_equals(response.status_code, 200, "Service update failed")
    print("✅ Service updated successfully!")


def test_delete_service(service_name: str):
    """Test service deletion."""
    print("\n🧪 Testing service deletion...")

    response = requests.delete(f"{BASE_URL}/services/{service_name}")
    assert_equals(
        response.status_code, 204, "Service deletion failed"
    )

    # Verify service is gone
    list_response = requests.get(f"{BASE_URL}/services/")
    services = list_response.json()
    for service in services:
        if service["name"] == service_name:
            raise AssertionError(
                f"Service {service_name} still exists after deletion"
            )

    print("✅ Service deleted successfully!")


def run_tests():
    """Run all tests in sequence."""
    try:
        print("🚀 Starting API tests...")

        # Run tests in sequence
        service_name = test_create_service()

        # Wait a bit for service to be fully ready
        print("⏳ Waiting for service to be ready...")
        time.sleep(5)

        test_list_services(service_name)
        test_update_service(service_name)
        test_delete_service(service_name)

        print("\n✨ All tests passed successfully! ✨")

    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise
    finally:
        print("\n🏁 Tests completed")


if __name__ == "__main__":
    run_tests()
