# Swarms Wallet API Documentation

This documentation covers the Swarms Wallet API routes for managing wallets, sending tokens, and checking transactions in the Swarms Platform.

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
Sends SWARMS tokens with automatic tax handling.

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
- SWARMS token tax: 2% from sender + 2% from sent amount
- All taxes are sent to the DAO treasury
- Token accounts are automatically created for new recipients
- Transactions use 'processed' commitment level


## Implementation Notes

- All token amounts should be provided in their natural units (not in lamports/raw units)
- SOL balances are returned in SOL (not lamports)
- Token accounts are automatically created for recipients if they don't exist
- All transactions include automatic tax handling for the DAO treasury
- Compute budget and priority fees are automatically managed for optimal transaction processing 
