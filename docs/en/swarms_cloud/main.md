# Swarm Cloud API Reference

## Overview

The AI Chat Completion API processes text and image inputs to generate conversational responses. It supports various configurations to customize response behavior and manage input content.

## API Endpoints

### Chat Completion URL
`https://api.swarms.world`



- **Endpoint:** `/v1/chat/completions`
-- **Full Url** `https://api.swarms.world/v1/chat/completions`
- **Method:** POST
- **Description:** Generates a response based on the provided conversation history and parameters.

#### Request Parameters

| Parameter     | Type               | Description                                               | Required |
|---------------|--------------------|-----------------------------------------------------------|----------|
| `model`       | string             | The AI model identifier.                                  | Yes      |
| `messages`    | array of objects   | A list of chat messages, including the sender's role and content. | Yes      |
| `temperature` | float              | Controls randomness. Lower values make responses more deterministic. | No       |
| `top_p`       | float              | Controls diversity. Lower values lead to less random completions. | No       |
| `max_tokens`  | integer            | The maximum number of tokens to generate.                 | No       |
| `stream`      | boolean            | If set to true, responses are streamed back as they're generated. | No       |

#### Response Structure

- **Success Response Code:** `200 OK`

```markdown
{
  "model": string,
  "object": string,
  "choices": array of objects,
  "usage": object
}
```

### List Models

- **Endpoint:** `/v1/models`
- **Method:** GET
- **Description:** Retrieves a list of available models.

#### Response Structure

- **Success Response Code:** `200 OK`

```markdown
{
  "data": array of objects
}
```

## Objects

### Request

| Field     | Type                | Description                                   | Required |
|-----------|---------------------|-----------------------------------------------|----------|
| `role`    | string              | The role of the message sender.               | Yes      |
| `content` | string or array     | The content of the message.                   | Yes      |
| `name`    | string              | An optional name identifier for the sender.   | No       |

### Response

| Field     | Type   | Description                        |
|-----------|--------|------------------------------------|
| `index`   | integer| The index of the choice.           |
| `message` | object | A `ChatMessageResponse` object.    |

#### UsageInfo

| Field             | Type    | Description                                   |
|-------------------|---------|-----------------------------------------------|
| `prompt_tokens`   | integer | The number of tokens used in the prompt.      |
| `total_tokens`    | integer | The total number of tokens used.              |
| `completion_tokens`| integer| The number of tokens used for the completion. |

## Example Requests

### Text Chat Completion

```json
POST /v1/chat/completions
{
  "model": "cogvlm-chat-17b",
  "messages": [
    {
      "role": "user",
      "content": "Hello, world!"
    }
  ],
  "temperature": 0.8
}
```

### Image and Text Chat Completion

```json
POST /v1/chat/completions
{
  "model": "cogvlm-chat-17b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image"
        },
        {
          "type": "image_url",
          "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
        }
      ]
    }
  ],
  "temperature": 0.8,
  "top_p": 0.9,
  "max_tokens": 1024
}
```

## Error Codes

The API uses standard HTTP status codes to indicate the success or failure of an API call.

| Status Code | Description                       |
|-------------|-----------------------------------|
| 200         | OK - The request has succeeded.   |
| 400         | Bad Request - Invalid request format. |
| 500         | Internal Server Error - An error occurred on the server. |


## Examples in Various Languages

### Python
```python
import requests
import base64
from PIL import Image
from io import BytesIO


# Convert image to Base64
def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Replace 'image.jpg' with the path to your image
base64_image = image_to_base64("your_image.jpg")
text_data = {"type": "text", "text": "Describe what is in the image"}
image_data = {
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
}

# Construct the request data
request_data = {
    "model": "cogvlm-chat-17b",
    "messages": [{"role": "user", "content": [text_data, image_data]}],
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 1024,
}

# Specify the URL of your FastAPI application
url = "https://api.swarms.world/v1/chat/completions"

# Send the request
response = requests.post(url, json=request_data)
# Print the response from the server
print(response.text)
```

### Example API Request in Node
```js
const fs = require('fs');
const https = require('https');
const sharp = require('sharp');

// Convert image to Base64
async function imageToBase64(imagePath) {
    try {
        const imageBuffer = await sharp(imagePath).jpeg().toBuffer();
        return imageBuffer.toString('base64');
    } catch (error) {
        console.error('Error converting image to Base64:', error);
    }
}

// Main function to execute the workflow
async function main() {
    const base64Image = await imageToBase64("your_image.jpg");
    const textData = { type: "text", text: "Describe what is in the image" };
    const imageData = {
        type: "image_url",
        image_url: { url: `data:image/jpeg;base64,${base64Image}` },
    };

    // Construct the request data
    const requestData = JSON.stringify({
        model: "cogvlm-chat-17b",
        messages: [{ role: "user", content: [textData, imageData] }],
        temperature: 0.8,
        top_p: 0.9,
        max_tokens: 1024,
    });

    const options = {
        hostname: 'api.swarms.world',
        path: '/v1/chat/completions',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': requestData.length,
        },
    };

    const req = https.request(options, (res) => {
        let responseBody = '';

        res.on('data', (chunk) => {
            responseBody += chunk;
        });

        res.on('end', () => {
            console.log('Response:', responseBody);
        });
    });

    req.on('error', (error) => {
        console.error(error);
    });

    req.write(requestData);
    req.end();
}

main();
```

### Example API Request in Go

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "image"
    "image/jpeg"
    _ "image/png" // Register PNG format
    "io"
    "net/http"
    "os"
)

// imageToBase64 converts an image to a Base64-encoded string.
func imageToBase64(imagePath string) (string, error) {
    file, err := os.Open(imagePath)
    if err != nil {
        return "", err
    }
    defer file.Close()

    img, _, err := image.Decode(file)
    if err != nil {
        return "", err
    }

    buf := new(bytes.Buffer)
    err = jpeg.Encode(buf, img, nil)
    if err != nil {
        return "", err
    }

    return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// main is the entry point of the program.
func main() {
    base64Image, err := imageToBase64("your_image.jpg")
    if err != nil {
        fmt.Println("Error converting image to Base64:", err)
        return
    }

    requestData := map[string]interface{}{
        "model": "cogvlm-chat-17b",
        "messages": []map[string]interface{}{
            {
                "role":    "user",
                "content": []map[string]string{{"type": "text", "text": "Describe what is in the image"}, {"type": "image_url", "image_url": {"url": fmt.Sprintf("data:image/jpeg;base64,%s", base64Image)}}},
            },
        },
        "temperature": 0.8,
        "top_p":       0.9,
        "max_tokens":  1024,
    }

    requestBody, err := json.Marshal(requestData)
    if err != nil {
        fmt.Println("Error marshaling request data:", err)
        return
    }

    url := "https://api.swarms.world/v1/chat/completions"
    request, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    request.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    response, err := client.Do(request)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer response.Body.Close()

    responseBody, err := io.ReadAll(response.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }

    fmt.Println("Response:", string(responseBody))
}
```





## Conclusion

This API reference provides the necessary details to understand and interact with the AI Chat Completion API. By following the outlined request and response formats, users can integrate this API into their applications to generate dynamic and contextually relevant conversational responses.