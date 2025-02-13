# swarms Wallet API Documentation

This documentation covers the swarms Wallet API routes for managing wallets, sending tokens, and checking transactions in the swarms Platform.

## Authentication

All endpoints require an API key to be passed in the request headers:

```http
x-api-key: your_api_key_here
```

## Endpoints

### Generate Wallet

Creates a new Solana wallet for an AI agent or retrieves an existing one.

```http
POST https://swarms.world/api/solana/generate-wallet
```

**Response**
```json
{
  "success": true,
  "data": {
    "public_key": "string",
    "wallet_type": "solana",
    "swarms_token_address": "string"
  },
  "code": "SUCCESS_001"
}
```

### Send Tokens
Sends swarms tokens with automatic tax handling.

```http
POST https://swarms.world/api/solana/send-tokens
```

**Request Body**
```json
{
  "recipientAddress": "string",
  "amount": "number",
  "solanaFee": "number" // Optional, default: 0.009
}
```

**Response**
```json
{
  "success": true,
  "data": {
    "signature": "string",
    "details": {
      "sender": "string",
      "recipient": "string",
      "daoAddress": "string",
      "requestedSendAmount": "number",
      "totalNeededFromAccount": "number",
      "accountTax": "number",
      "receivedTax": "number",
      "recipientReceives": "number",
      "taxBreakdown": "string",
      "computeUnits": "number",
      "priorityFee": "number"
    }
  },
  "code": "SUCCESS_001"
}
```

### Check Receipt
Verifies token receipt and checks balances.

```http
GET https://swarms.world/api/solana/check-receipt?amount={amount}
```

**Response**
```json
{
  "success": true,
  "data": {
    "solana_address": "string",
    "received": "number",
    "expected": "number",
    "matches": "boolean",
    "balances": {
      "sol": "number",
      "swarms": "number"
    },
    "swarms_address": "string"
  },
  "code": "SUCCESS_001"
}
```

### Get Metrics
Retrieves transaction metrics and history.

```http
GET https://swarms.world/api/solana/get-metrics
```

**Query Parameters**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 10, max: 100)
- `startDate`: Filter start date
- `endDate`: Filter end date
- `status`: Transaction status filter
- `type`: Transaction type filter

**Response**
```json
{
  "success": true,
  "data": {
    "transactions": [{
      "id": "string",
      "agent_id": "string",
      "transaction_hash": "string",
      "amount": "number",
      "recipient": "string",
      "status": "string",
      "transaction_type": "string",
      "created_at": "string"
    }],
    "pagination": {
      "currentPage": "number",
      "totalPages": "number",
      "totalItems": "number",
      "itemsPerPage": "number",
      "hasMore": "boolean"
    },
    "metrics": {
      "totalTransactions": "number",
      "totalAmountSent": "number",
      "totalSuccessfulTransactions": "number",
      "totalFailedTransactions": "number"
    }
  },
  "code": "SUCCESS_001"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| AUTH_001 | Missing API key |
| AUTH_002 | Invalid API key |
| BAL_001 | Insufficient SOL balance |
| BAL_002 | Insufficient token balance |
| WAL_001 | Wallet not found |
| REQ_001 | Missing required parameters |
| DB_001 | Database error |
| ERR_001 | Internal server error |

## Transaction Details

- Default SOL fee: 0.009 SOL
- swarms token tax: 2% from sender + 2% from sent amount
- All taxes are sent to the DAO treasury
- Token accounts are automatically created for new recipients
- Transactions use 'processed' commitment level


## Implementation Notes

- All token amounts should be provided in their natural units (not in lamports/raw units)
- SOL balances are returned in SOL (not lamports)
- Token accounts are automatically created for recipients if they don't exist
- All transactions include automatic tax handling for the DAO treasury
- Compute budget and priority fees are automatically managed for optimal transaction processing 



## Examples

Below are code examples in several languages that demonstrate how to use the swarms Wallet API endpoints. In these examples, replace `your_api_key_here` with your actual API key, and update any parameters as needed.

---

## Python (Using `requests`)

First, install the library if you haven’t already:

```bash
pip install requests
```

**Example: Generate Wallet**

```python
import os
import requests

API_KEY = os.getenv("SWARMS_API_KEY")
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

url = "https://swarms.world/api/solana/generate-wallet"
response = requests.post(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Wallet generated:", data)
else:
    print("Error:", response.text)
```

**Example: Send Tokens**

```python
import requests
import json
import os

API_KEY = os.getenv("SWARMS_API_KEY")
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

url = "https://swarms.world/api/solana/send-tokens"
payload = {
    "recipientAddress": "recipient_public_key",
    "amount": 100,  # Example token amount
    # "solanaFee": 0.009  # Optional: use default if not provided
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
if response.status_code == 200:
    data = response.json()
    print("Tokens sent:", data)
else:
    print("Error:", response.text)
```

**Example: Check Receipt**

```python
import requests
import os

API_KEY = os.getenv("SWARMS_API_KEY")
headers = {
    "x-api-key": API_KEY
}

amount = 100  # The amount you expect to be received
url = f"https://swarms.world/api/solana/check-receipt?amount={amount}"

response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    print("Receipt checked:", data)
else:
    print("Error:", response.text)
```

**Example: Get Metrics**

```python
import requests
import os

API_KEY = os.getenv("SWARMS_API_KEY")
headers = {
    "x-api-key": API_KEY
}

params = {
    "page": 1,
    "limit": 10,
    # Optionally include startDate, endDate, status, type if needed.
}

url = "https://swarms.world/api/solana/get-metrics"
response = requests.get(url, headers=headers, params=params)
if response.status_code == 200:
    data = response.json()
    print("Metrics:", data)
else:
    print("Error:", response.text)
```

---

## Node.js (Using `axios`)

First, install axios:

```bash
npm install axios
```

**Example: Generate Wallet**

```javascript
const axios = require('axios');

const API_KEY = 'your_api_key_here';
const headers = {
  'x-api-key': API_KEY,
  'Content-Type': 'application/json'
};

axios.post('https://swarms.world/api/solana/generate-wallet', {}, { headers })
  .then(response => {
    console.log('Wallet generated:', response.data);
  })
  .catch(error => {
    console.error('Error:', error.response ? error.response.data : error.message);
  });
```

**Example: Send Tokens**

```javascript
const axios = require('axios');

const API_KEY = 'your_api_key_here';
const headers = {
  'x-api-key': API_KEY,
  'Content-Type': 'application/json'
};

const payload = {
  recipientAddress: 'recipient_public_key',
  amount: 100, // token amount
  // solanaFee: 0.009 // Optional
};

axios.post('https://swarms.world/api/solana/send-tokens', payload, { headers })
  .then(response => {
    console.log('Tokens sent:', response.data);
  })
  .catch(error => {
    console.error('Error:', error.response ? error.response.data : error.message);
  });
```

**Example: Check Receipt**

```javascript
const axios = require('axios');

const API_KEY = 'your_api_key_here';
const headers = { 'x-api-key': API_KEY };
const amount = 100;
const url = `https://swarms.world/api/solana/check-receipt?amount=${amount}`;

axios.get(url, { headers })
  .then(response => {
    console.log('Receipt:', response.data);
  })
  .catch(error => {
    console.error('Error:', error.response ? error.response.data : error.message);
  });
```

**Example: Get Metrics**

```javascript
const axios = require('axios');

const API_KEY = 'your_api_key_here';
const headers = { 'x-api-key': API_KEY };

const params = {
  page: 1,
  limit: 10,
  // startDate: '2025-01-01', endDate: '2025-01-31', status: 'completed', type: 'send'
};

axios.get('https://swarms.world/api/solana/get-metrics', { headers, params })
  .then(response => {
    console.log('Metrics:', response.data);
  })
  .catch(error => {
    console.error('Error:', error.response ? error.response.data : error.message);
  });
```

---

## cURL (Command Line)

**Example: Generate Wallet**

```bash
curl -X POST https://swarms.world/api/solana/generate-wallet \
  -H "x-api-key: your_api_key_here" \
  -H "Content-Type: application/json"
```

**Example: Send Tokens**

```bash
curl -X POST https://swarms.world/api/solana/send-tokens \
  -H "x-api-key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "recipientAddress": "recipient_public_key",
    "amount": 100,
    "solanaFee": 0.009
  }'
```

**Example: Check Receipt**

```bash
curl -X GET "https://swarms.world/api/solana/check-receipt?amount=100" \
  -H "x-api-key: your_api_key_here"
```

**Example: Get Metrics**

```bash
curl -X GET "https://swarms.world/api/solana/get-metrics?page=1&limit=10" \
  -H "x-api-key: your_api_key_here"
```

---

## Other Languages

### Ruby (Using `net/http`)

**Example: Generate Wallet**

```ruby
require 'net/http'
require 'uri'
require 'json'

uri = URI.parse("https://swarms.world/api/solana/generate-wallet")
request = Net::HTTP::Post.new(uri)
request["x-api-key"] = "your_api_key_here"
request["Content-Type"] = "application/json"

response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
  http.request(request)
end

puts JSON.parse(response.body)
```

### Java (Using `HttpURLConnection`)

**Example: Generate Wallet**

```java
import java.io.*;
import java.net.*;
import javax.net.ssl.HttpsURLConnection;

public class SwarmsApiExample {
    public static void main(String[] args) {
        try {
            URL url = new URL("https://swarms.world/api/solana/generate-wallet");
            HttpsURLConnection conn = (HttpsURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("x-api-key", "your_api_key_here");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);

            // If you need to send a request body, write to the output stream:
            // try(OutputStream os = conn.getOutputStream()) {
            //     byte[] input = "{}".getBytes("utf-8");
            //     os.write(input, 0, input.length);
            // }

            BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"));
            StringBuilder response = new StringBuilder();
            String responseLine = null;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            System.out.println("Response: " + response.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

---

These examples illustrate how to authenticate using the API key and perform various operations such as generating a wallet, sending tokens, checking receipts, and retrieving metrics. You can adapt these examples to other languages or frameworks as needed. Enjoy integrating with the swarms Wallet API!