# Prompts API Documentation

The `https://swarms.world/api/add-prompt` endpoint allows users to add a new prompt to the Swarms platform. This API accepts a POST request with a JSON body containing details of the prompt, such as its name, description, use cases, and tags. The request must be authenticated using an API key.

## Endpoint: Add Prompt

- **URL:** `https://swarms.world/api/add-prompt`
- **Method:** POST
- **Content-Type:** `application/json`
- **Authorization:** Bearer token required in the header

## Request Parameters

The request body should be a JSON object with the following attributes:

| Attribute     | Type     | Description                                                     | Required |
| ------------- | -------- | --------------------------------------------------------------- | -------- |
| `name`        | `string` | The name of the prompt.                                         | Yes      |
| `prompt`      | `string` | The prompt text.                                                | Yes      |
| `description` | `string` | A brief description of the prompt.                              | Yes      |
| `useCases`    | `array`  | An array of use cases, each containing a title and description. | Yes      |
| `tags`        | `string` | Comma-separated tags for the prompt.                            | No       |

### `useCases` Structure

Each use case in the `useCases` array should be an object with the following attributes:

| Attribute     | Type     | Description                          | Required |
| ------------- | -------- | ------------------------------------ | -------- |
| `title`       | `string` | The title of the use case.           | Yes      |
| `description` | `string` | A brief description of the use case. | Yes      |

## Example Usage

### Python

```python
import requests
import json

url = "https://swarms.world/api/add-prompt"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {apiKey}"
}
data = {
    "name": "Example Prompt",
    "prompt": "This is an example prompt from an API route.",
    "description": "Description of the prompt.",
    "useCases": [
        {"title": "Use case 1", "description": "Description of use case 1"},
        {"title": "Use case 2", "description": "Description of use case 2"}
    ],
    "tags": "example, prompt"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

### Node.js

```javascript
const fetch = require("node-fetch");

async function addPromptsHandler() {
  try {
    const response = await fetch("https://swarms.world/api/add-prompt", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer {apiKey}",
      },
      body: JSON.stringify({
        name: "Example Prompt",
        prompt: "This is an example prompt from an API route.",
        description: "Description of the prompt.",
        useCases: [
          { title: "Use case 1", description: "Description of use case 1" },
          { title: "Use case 2", description: "Description of use case 2" },
        ],
        tags: "example, prompt",
      }),
    });

    const result = await response.json();
    console.log(result);
  } catch (error) {
    console.error("An error has occurred", error);
  }
}

addPromptsHandler();
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    url := "https://swarms.world/api/add-prompt"
    payload := map[string]interface{}{
        "name":        "Example Prompt",
        "prompt":      "This is an example prompt from an API route.",
        "description": "Description of the prompt.",
        "useCases": []map[string]string{
            {"title": "Use case 1", "description": "Description of use case 1"},
            {"title": "Use case 2", "description": "Description of use case 2"},
        },
        "tags": "example, prompt",
    }
    jsonPayload, _ := json.Marshal(payload)

    req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer {apiKey}")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        fmt.Println("An error has occurred", err)
        return
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    fmt.Println(result)
}
```

### cURL

```bash
curl -X POST https://swarms.world/api/add-prompt \
-H "Content-Type: application/json" \
-H "Authorization: Bearer {apiKey}" \
-d '{
  "name": "Example Prompt",
  "prompt": "This is an example prompt from an API route.",
  "description": "Description of the prompt.",
  "useCases": [
    { "title": "Use case 1", "description": "Description of use case 1" },
    { "title": "Use case 2", "description": "Description of use case 2" }
  ],
  "tags": "example, prompt"
}'
```

## Response

The response will be a JSON object containing the result of the operation. Example response:

```json
{
  "success": true,
  "message": "Prompt added successfully",
  "data": {
    "id": "prompt_id",
    "name": "Example Prompt",
    "prompt": "This is an example prompt from an API route.",
    "description": "Description of the prompt.",
    "useCases": [
      { "title": "Use case 1", "description": "Description of use case 1" },
      { "title": "Use case 2", "description": "Description of use case 2" }
    ],
    "tags": "example, prompt"
  }
}
```