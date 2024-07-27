# Documentation for `getAllPrompts` API Endpoint

The `getAllPrompts` API endpoint is a part of the `swarms.world` application, designed to fetch all prompt records from the database. This endpoint is crucial for retrieving various prompts stored in the `swarms_cloud_prompts` table, including their metadata such as name, description, use cases, and tags. It provides an authenticated way to access this data, ensuring that only authorized users can retrieve the information.

## Purpose

The primary purpose of this API endpoint is to provide a secure method for clients to fetch a list of prompts stored in the `swarms_cloud_prompts` table. It ensures data integrity and security by using an authentication guard and handles various HTTP methods and errors gracefully.

## API Endpoint Definition

### Endpoint URL

```
https://swarms.world/get-prompt
```

### HTTP Method

```
GET
```

### Request Headers

| Header         | Type   | Required | Description                 |
|----------------|--------|----------|-----------------------------|
| Authorization  | String | Yes      | Bearer token for API access |

### Response

#### Success Response (200)

Returns an array of prompts.

```json
[
  {
    "id": "string",
    "name": "string",
    "description": "string",
    "prompt": "string",
    "use_cases": "string",
    "tags": "string"
  },
  ...
]
```

#### Error Responses

- **405 Method Not Allowed**

  ```json
  {
    "error": "Method <method> Not Allowed"
  }
  ```

- **401 Unauthorized**

  ```json
  {
    "error": "API Key is missing"
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "Could not fetch prompts"
  }
  ```

## Request Handling

1. **Method Validation**: The endpoint only supports the `GET` method. If a different HTTP method is used, it responds with a `405 Method Not Allowed` status.
2. **Authentication**: The request must include a valid Bearer token in the `Authorization` header. If the token is missing, a `401 Unauthorized` status is returned.
3. **Authorization Check**: The token is validated using the `AuthApiGuard` class. If the token is invalid, an appropriate error response is returned based on the status received from the guard.
4. **Database Query**: The endpoint queries the `swarms_cloud_prompts` table using the `supabaseAdmin` client to fetch prompt data.
5. **Response**: On success, it returns the prompt data in JSON format. In case of an error during the database query, a `500 Internal Server Error` status is returned.

## Code Example

### JavaScript (Node.js)

```javascript
import fetch from 'node-fetch';

const getPrompts = async () => {
  const response = await fetch('https://swarms.world/get-prompt', {
    method: 'GET',
    headers: {
      'Authorization': 'Bearer YOUR_API_KEY'
    }
  });

  if (!response.ok) {
    throw new Error(`Error: ${response.statusText}`);
  }

  const data = await response.json();
  console.log(data);
};

getPrompts().catch(console.error);
```

### Python

```python
import requests

def get_prompts():
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY'
    }
    response = requests.get('https://swarms.world/get-prompt', headers=headers)

    if response.status_code != 200:
        raise Exception(f'Error: {response.status_code}, {response.text}')

    data = response.json()
    print(data)

get_prompts()
```

### cURL

```sh
curl -X GET https://swarms.world/get-prompt \
-H "Authorization: Bearer YOUR_API_KEY"
```

### Go

```go
package main

import (
    "fmt"
    "io/ioutil"
    "net/http"
)

func getPrompts() {
    client := &http.Client{}
    req, err := http.NewRequest("GET", "https://swarms.world/get-prompt", nil)
    if err != nil {
        panic(err)
    }

    req.Header.Add("Authorization", "Bearer YOUR_API_KEY")

    resp, err := client.Do(req)
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := ioutil.ReadAll(resp.Body)
        panic(fmt.Sprintf("Error: %d, %s", resp.StatusCode, string(body)))
    }

    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        panic(err)
    }

    fmt.Println(string(body))
}

func main() {
    getPrompts()
}
```

### Attributes Table

| Attribute     | Type   | Description                                                |
|---------------|--------|------------------------------------------------------------|
| id            | String | Unique identifier for the prompt                            |
| name          | String | Name of the prompt                                          |
| description   | String | Description of the prompt                                   |
| prompt        | String | The actual prompt text                                      |
| use_cases     | String | Use cases for the prompt                                    |
| tags          | String | Tags associated with the prompt                             |

## Additional Information and Tips

- Ensure your API key is kept secure and not exposed in client-side code.
- Handle different error statuses appropriately to provide clear feedback to users.
- Consider implementing rate limiting and logging for better security and monitoring.

## References and Resources

- [Next.js API Routes](https://nextjs.org/docs/api-routes/introduction)
- [Supabase Documentation](https://supabase.com/docs)
- [Node Fetch](https://www.npmjs.com/package/node-fetch)
- [Requests Library (Python)](https://docs.python-requests.org/en/latest/)
- [Go net/http Package](https://pkg.go.dev/net/http)

This documentation provides a comprehensive guide to the `getAllPrompts` API endpoint, including usage examples in multiple programming languages and detailed attribute descriptions.