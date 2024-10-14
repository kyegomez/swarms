#!/bin/bash

# Define the base URL
base_url="http://localhost:8000"

# Define the JSON payload
payload='{"feature": "login system", "codebase": "existing codebase here"}'

# Send POST request
echo "Sending request to /agent/ endpoint..."
response=$(curl -s -X POST "$base_url/agent/" -H "Content-Type: application/json" -d "$payload")

echo "Response: $response"