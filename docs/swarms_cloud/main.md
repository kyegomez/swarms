# Swarm Cloud API Reference

## Overview

The AI Chat Completion API processes text and image inputs to generate conversational responses. It supports various configurations to customize response behavior and manage input content.

## API Endpoints

### Chat Completion
`https://swarms.world`

- **Endpoint:** `/v1/chat/completions`
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

### ChatMessageInput

| Field     | Type                | Description                                   | Required |
|-----------|---------------------|-----------------------------------------------|----------|
| `role`    | string              | The role of the message sender.               | Yes      |
| `content` | string or array     | The content of the message.                   | Yes      |
| `name`    | string              | An optional name identifier for the sender.   | No       |

### ChatCompletionResponseChoice

| Field     | Type   | Description                        |
|-----------|--------|------------------------------------|
| `index`   | integer| The index of the choice.           |
| `message` | object | A `ChatMessageResponse` object.    |

### UsageInfo

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

## Conclusion

This API reference provides the necessary details to understand and interact with the AI Chat Completion API. By following the outlined request and response formats, users can integrate this API into their applications to generate dynamic and contextually relevant conversational responses.