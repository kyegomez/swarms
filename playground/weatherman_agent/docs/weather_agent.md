## Weather Agent API Documentation

### Overview
The Weather Agent API provides endpoints to interact with a weather prediction model, "WeatherMan Agent". This API allows users to get weather-related information through chat completions using the OpenAI GPT model with specific prompts and tools.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check

##### `GET /v1/health`
Checks the health status of the API.

**Response:**
- `200 OK`: Returns a JSON object indicating the status of the API.
  ```json
  {
    "status": "ok"
  }
  ```

#### Get Models

##### `GET /v1/models`
Retrieves the list of available models.

**Response:**
- `200 OK`: Returns a JSON object with the list of models.
  ```json
  {
    "models": ["WeatherMan Agent"]
  }
  ```

#### Chat Completions

##### `POST /v1/chat/completions`
Generates weather-related responses based on the provided prompt using the "WeatherMan Agent" model.

**Request Body:**
- `model` (string): The name of the model to use. Must be "WeatherMan Agent".
- `prompt` (string): The input prompt for the chat completion.
- `max_tokens` (integer, optional): The maximum number of tokens to generate. Default is 100.
- `temperature` (float, optional): The sampling temperature for the model. Default is 1.0.

**Example Request:**
```json
{
  "model": "WeatherMan Agent",
  "prompt": "What will the weather be like tomorrow in New York?",
  "max_tokens": 100,
  "temperature": 1.0
}
```

**Response:**
- `200 OK`: Returns a JSON object with the completion result.
  ```json
  {
    "id": "unique-id",
    "object": "text_completion",
    "created": 1234567890,
    "model": "WeatherMan Agent",
    "choices": [
      {
        "text": "The weather tomorrow in New York will be..."
      }
    ],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 15,
      "total_tokens": 25
    }
  }
  ```
- `400 Bad Request`: If the model specified is not "WeatherMan Agent".
  ```json
  {
    "detail": "Model not found"
  }
  ```
- `500 Internal Server Error`: If there is an error processing the request.
  ```json
  {
    "detail": "Error message"
  }
  ```

### Models
The API supports the following model:
- **WeatherMan Agent**: A specialized agent for providing weather-related information based on the prompt.

### Usage

1. **Health Check:** Verify that the API is running by sending a GET request to `/v1/health`.
2. **Get Models:** Retrieve the list of available models by sending a GET request to `/v1/models`.
3. **Chat Completions:** Generate a weather-related response by sending a POST request to `/v1/chat/completions` with the required parameters.

### Error Handling
The API returns appropriate HTTP status codes and error messages for different error scenarios:
- `400 Bad Request` for invalid requests.
- `500 Internal Server Error` for unexpected errors during processing.

### CORS Configuration
The API allows cross-origin requests from any origin, supporting all methods and headers.

---

For further assistance or issues, please contact the API support team.