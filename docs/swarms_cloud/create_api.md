# CreateNow API Documentation

Welcome to the CreateNow API documentation! This API enables developers to generate AI-powered content, including images, music, videos, and speech, using natural language prompts. Use the endpoints below to start generating content.

---

## **1. Claim Your API Key**
To use the API, you must first claim your API key. Visit the following link to create an account and get your API key:

### **Claim Your Key**
```
https://createnow.xyz/account
```

After signing up, your API key will be available in your account dashboard. Keep it secure and include it in your API requests as a Bearer token.

---

## **2. Generation Endpoint**
The generation endpoint allows you to create AI-generated content using natural language prompts.

### **Endpoint**
```
POST https://createnow.xyz/api/v1/generate
```

### **Authentication**
Include a Bearer token in the `Authorization` header for all requests:
```
Authorization: Bearer YOUR_API_KEY
```

### **Basic Usage**
The simplest way to use the API is to send a prompt. The system will automatically detect the appropriate media type.

#### **Example Request (Basic)**
```json
{
  "prompt": "a beautiful sunset over the ocean"
}
```

### **Advanced Options**
You can specify additional parameters for finer control over the output.

#### **Parameters**
| Parameter      | Type      | Description                                                                                       | Default      |
|----------------|-----------|---------------------------------------------------------------------------------------------------|--------------|
| `prompt`       | `string`  | The natural language description of the content to generate.                                     | Required     |
| `type`         | `string`  | The type of content to generate (`image`, `music`, `video`, `speech`).                          | Auto-detect  |
| `count`        | `integer` | The number of outputs to generate (1-4).                                                        | 1            |
| `duration`     | `integer` | Duration of audio or video content in seconds (applicable to `music` and `speech`).            | N/A          |

#### **Example Request (Advanced)**
```json
{
  "prompt": "create an upbeat jazz melody",
  "type": "music",
  "count": 2,
  "duration": 30
}
```

### **Response Format**

#### **Success Response**
```json
{
  "success": true,
  "outputs": [
    {
      "url": "https://createnow.xyz/storage/image1.png",
      "creation_id": "12345",
      "share_url": "https://createnow.xyz/share/12345"
    }
  ],
  "mediaType": "image",
  "confidence": 0.95,
  "detected": true
}
```

#### **Error Response**
```json
{
  "error": "Invalid API Key",
  "status": 401
}
```

---

## **3. Examples in Multiple Languages**

### **Python**
```python
import requests

url = "https://createnow.xyz/api/v1/generate"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "prompt": "a futuristic cityscape at night",
    "type": "image",
    "count": 2
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### **Node.js**
```javascript
const axios = require('axios');

const url = "https://createnow.xyz/api/v1/generate";
const headers = {
    Authorization: "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
};

const payload = {
    prompt: "a futuristic cityscape at night",
    type: "image",
    count: 2
};

axios.post(url, payload, { headers })
    .then(response => {
        console.log(response.data);
    })
    .catch(error => {
        console.error(error.response.data);
    });
```

### **cURL**
```bash
curl -X POST https://createnow.xyz/api/v1/generate \
-H "Authorization: Bearer YOUR_API_KEY" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "a futuristic cityscape at night",
  "type": "image",
  "count": 2
}'
```

### **Java**
```java
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;

public class CreateNowAPI {
    public static void main(String[] args) throws Exception {
        URL url = new URL("https://createnow.xyz/api/v1/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Authorization", "Bearer YOUR_API_KEY");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);

        String jsonPayload = "{" +
            "\"prompt\": \"a futuristic cityscape at night\", " +
            "\"type\": \"image\", " +
            "\"count\": 2}";

        OutputStream os = conn.getOutputStream();
        os.write(jsonPayload.getBytes());
        os.flush();

        int responseCode = conn.getResponseCode();
        System.out.println("Response Code: " + responseCode);
    }
}
```

---

## **4. Error Codes**
| Status Code | Meaning                          | Possible Causes                        |
|-------------|----------------------------------|----------------------------------------|
| 400         | Bad Request                      | Invalid parameters or payload.         |
| 401         | Unauthorized                     | Invalid or missing API key.            |
| 402         | Payment Required                 | Insufficient credits for the request.  |
| 500         | Internal Server Error            | Issue on the server side.              |

---

## **5. Notes and Limitations**
- **Maximum Prompt Length:** 1000 characters.
- **Maximum Outputs per Request:** 4.
- **Supported Media Types:** `image`, `music`, `video`, `speech`.
- **Content Shareability:** Every output includes a unique creation ID and shareable URL.
- **Auto-Detection:** Uses advanced natural language processing to determine the most appropriate media type.

---

For further support or questions, please contact our support team at [support@createnow.xyz](mailto:support@createnow.xyz).

