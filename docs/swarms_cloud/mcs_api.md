# Medical Coder Swarm API Documentation

Base URL: `https://mcs-285321057562.us-central1.run.app`

## Table of Contents
- [Authentication](#authentication)
- [Rate Limits](#rate-limits)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Run Medical Coder](#run-medical-coder)
  - [Run Batch Medical Coder](#run-batch-medical-coder)
  - [Get Patient Data](#get-patient-data)
  - [Get All Patients](#get-all-patients)
- [Code Examples](#code-examples)
- [Error Handling](#error-handling)

## Authentication

Authentication details will be provided by the MCS team. Contact support for API credentials.

## Rate Limits

| Endpoint | GET Rate Limit Status |
|----------|----------------------|
| `GET /rate-limits` | Returns current rate limit status for your IP address |

## Endpoints

### Health Check

Check if the API is operational.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns 200 OK if service is running |

### Run Medical Coder

Process a single patient case through the Medical Coder Swarm.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/medical-coder/run` | Process a single patient case |

**Request Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| patient_id | string | Yes | Unique identifier for the patient |
| case_description | string | Yes | Medical case details to be processed |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| patient_id | string | Patient identifier |
| case_data | string | Processed case data |

### Run Batch Medical Coder

Process multiple patient cases in a single request.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/medical-coder/run-batch` | Process multiple patient cases |

**Request Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| cases | array | Yes | Array of PatientCase objects |

### Get Patient Data

Retrieve data for a specific patient.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/medical-coder/patient/{patient_id}` | Get patient data by ID |

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| patient_id | string | Yes | Patient identifier |

### Get All Patients

Retrieve data for all patients.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/medical-coder/patients` | Get all patient data |

## Code Examples

### Python

```python
import requests
import json

class MCSClient:
    def __init__(self, base_url="https://mcs.swarms.ai", api_key=None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else None
        }

    def run_medical_coder(self, patient_id, case_description):
        endpoint = f"{self.base_url}/v1/medical-coder/run"
        payload = {
            "patient_id": patient_id,
            "case_description": case_description
        }
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()

    def run_batch(self, cases):
        endpoint = f"{self.base_url}/v1/medical-coder/run-batch"
        payload = {"cases": cases}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()

# Usage example
client = MCSClient(api_key="your_api_key")
result = client.run_medical_coder("P123", "Patient presents with...")
```

### Next.js (TypeScript)

```typescript
// types.ts
interface PatientCase {
  patient_id: string;
  case_description: string;
}

interface QueryResponse {
  patient_id: string;
  case_data: string;
}

// api.ts
export class MCSApi {
  private baseUrl: string;
  private apiKey: string;

  constructor(apiKey: string, baseUrl = 'https://mcs.swarms.ai') {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async fetchWithAuth(endpoint: string, options: RequestInit = {}) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
        ...options.headers,
      },
    });
    return response.json();
  }

  async runMedicalCoder(patientCase: PatientCase): Promise<QueryResponse> {
    return this.fetchWithAuth('/v1/medical-coder/run', {
      method: 'POST',
      body: JSON.stringify(patientCase),
    });
  }

  async getPatientData(patientId: string): Promise<QueryResponse> {
    return this.fetchWithAuth(`/v1/medical-coder/patient/${patientId}`);
  }
}

// Usage in component
const mcsApi = new MCSApi(process.env.MCS_API_KEY);

export async function ProcessPatientCase({ patientId, caseDescription }) {
  const result = await mcsApi.runMedicalCoder({
    patient_id: patientId,
    case_description: caseDescription,
  });
  return result;
}
```

### Go

```go
package mcs

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type MCSClient struct {
    BaseURL string
    APIKey  string
    Client  *http.Client
}

type PatientCase struct {
    PatientID      string `json:"patient_id"`
    CaseDescription string `json:"case_description"`
}

type QueryResponse struct {
    PatientID string `json:"patient_id"`
    CaseData  string `json:"case_data"`
}

func NewMCSClient(apiKey string) *MCSClient {
    return &MCSClient{
        BaseURL: "https://mcs.swarms.ai",
        APIKey:  apiKey,
        Client:  &http.Client{},
    }
}

func (c *MCSClient) RunMedicalCoder(patientCase PatientCase) (*QueryResponse, error) {
    payload, err := json.Marshal(patientCase)
    if err != nil {
        return nil, err
    }

    req, err := http.NewRequest("POST", 
        fmt.Sprintf("%s/v1/medical-coder/run", c.BaseURL),
        bytes.NewBuffer(payload))
    if err != nil {
        return nil, err
    }

    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.APIKey))

    resp, err := c.Client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result QueryResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return &result, nil
}

// Usage example
func main() {
    client := NewMCSClient("your_api_key")
    
    result, err := client.RunMedicalCoder(PatientCase{
        PatientID:      "P123",
        CaseDescription: "Patient presents with...",
    })
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Result: %+v\n", result)
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages in JSON format.

**Common Status Codes:**

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid or missing API key |
| 422 | Validation Error - Request validation failed |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

**Error Response Format:**

```json
{
  "detail": [
    {
      "loc": ["body", "patient_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```


# MCS Python Client Documentation

## Installation

```bash
pip install mcs
```

## Quick Start

```python
from mcs import MCSClient, PatientCase

# Using context manager (recommended)
with MCSClient() as client:
    # Process a single case
    response = client.run_medical_coder(
        patient_id="P123",
        case_description="Patient presents with acute respiratory symptoms..."
    )
    print(f"Processed case: {response.case_data}")

    # Process multiple cases
    cases = [
        PatientCase("P124", "Case 1 description..."),
        PatientCase("P125", "Case 2 description...")
    ]
    batch_response = client.run_batch(cases)
```

## Client Configuration

### Constructor Arguments

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| api_key | str | Yes | - | Authentication API key |
| base_url | str | No | "https://mcs.swarms.ai" | API base URL |
| timeout | int | No | 30 | Request timeout in seconds |
| max_retries | int | No | 3 | Maximum retry attempts |
| logger_name | str | No | "mcs" | Name for the logger instance |

### Example Configuration

```python
client = MCSClient(
    base_url="https://custom-url.example.com",
    timeout=45,
    max_retries=5,
    logger_name="custom_logger"
)
```

## Data Models

### PatientCase

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| patient_id | str | Yes | Unique identifier for the patient |
| case_description | str | Yes | Medical case details |

### QueryResponse

| Field | Type | Description |
|-------|------|-------------|
| patient_id | str | Patient identifier |
| case_data | str | Processed case data |

## Methods

### run_medical_coder

Process a single patient case.

```python
def run_medical_coder(
    self,
    patient_id: str,
    case_description: str
) -> QueryResponse:
```

**Arguments:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| patient_id | str | Yes | Patient identifier |
| case_description | str | Yes | Case details |

**Example:**
```python
response = client.run_medical_coder(
    patient_id="P123",
    case_description="Patient presents with..."
)
print(response.case_data)
```

### run_batch

Process multiple patient cases in batch.

```python
def run_batch(
    self,
    cases: List[PatientCase]
) -> List[QueryResponse]:
```

**Arguments:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| cases | List[PatientCase] | Yes | List of patient cases |

**Example:**
```python
cases = [
    PatientCase("P124", "Case 1 description..."),
    PatientCase("P125", "Case 2 description...")
]
responses = client.run_batch(cases)
for response in responses:
    print(f"Patient {response.patient_id}: {response.case_data}")
```

### get_patient_data

Retrieve data for a specific patient.

```python
def get_patient_data(
    self,
    patient_id: str
) -> QueryResponse:
```

**Example:**
```python
patient_data = client.get_patient_data("P123")
print(f"Patient data: {patient_data.case_data}")
```

### get_all_patients

Retrieve data for all patients.

```python
def get_all_patients(self) -> List[QueryResponse]:
```

**Example:**
```python
all_patients = client.get_all_patients()
for patient in all_patients:
    print(f"Patient {patient.patient_id}: {patient.case_data}")
```

### get_rate_limits

Get current rate limit status.

```python
def get_rate_limits(self) -> Dict[str, Any]:
```

**Example:**
```python
rate_limits = client.get_rate_limits()
print(f"Rate limit status: {rate_limits}")
```

### health_check

Check if the API is operational.

```python
def health_check(self) -> bool:
```

**Example:**
```python
is_healthy = client.health_check()
print(f"API health: {'Healthy' if is_healthy else 'Unhealthy'}")
```

## Error Handling

### Exception Hierarchy

| Exception | Description |
|-----------|-------------|
| MCSClientError | Base exception for all client errors |
| RateLimitError | Raised when API rate limit is exceeded |
| AuthenticationError | Raised when API authentication fails |
| ValidationError | Raised when request validation fails |

### Example Error Handling

```python
from mcs import MCSClient, MCSClientError, RateLimitError

with MCSClient() as client:
    try:
        response = client.run_medical_coder("P123", "Case description...")
    except RateLimitError:
        print("Rate limit exceeded. Please wait before retrying.")
    except MCSClientError as e:
        print(f"An error occurred: {str(e)}")
```

## Advanced Usage

### Retry Configuration

The client implements two levels of retry logic:

1. Connection-level retries (using `HTTPAdapter`):
```python
client = MCSClient(
    ,
    max_retries=5  # Adjusts connection-level retries
)
```

2. Application-level retries (using `tenacity`):
```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(5))
def process_with_custom_retries():
    with MCSClient() as client:
        return client.run_medical_coder("P123", "Case description...")
```

### Batch Processing with Progress Tracking

```python
from tqdm import tqdm

with MCSClient() as client:
    cases = [
        PatientCase(f"P{i}", f"Case description {i}")
        for i in range(100)
    ]
    
    # Process in smaller batches
    batch_size = 10
    results = []
    
    for i in tqdm(range(0, len(cases), batch_size)):
        batch = cases[i:i + batch_size]
        batch_results = client.run_batch(batch)
        results.extend(batch_results)
```

## Best Practices

1. **Always use context managers:**
   ```python
   with MCSClient() as client:
       # Your code here
       pass
   ```

2. **Handle rate limits appropriately:**
   ```python
   from time import sleep
   
   def process_with_rate_limit_handling():
       with MCSClient() as client:
           try:
               return client.run_medical_coder("P123", "Case...")
           except RateLimitError:
               sleep(60)  # Wait before retry
               return client.run_medical_coder("P123", "Case...")
   ```

3. **Implement proper logging:**
   ```python
   from loguru import logger
   
   logger.add("mcs.log", rotation="500 MB")
   
   with MCSClient() as client:
       try:
           response = client.run_medical_coder("P123", "Case...")
       except Exception as e:
           logger.exception(f"Error processing case: {str(e)}")
   ```

4. **Monitor API health:**
   ```python
   def ensure_healthy_api():
       with MCSClient() as client:
           if not client.health_check():
               raise SystemExit("API is not healthy")
   ```