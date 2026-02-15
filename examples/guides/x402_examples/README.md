# X402 Research Agent Example

A simple example showing how to build a research agent with Swarms and monetize it using the x402 payment framework.

## What is x402?

x402 is a payment protocol that enables you to monetize your AI agents and APIs with crypto payments. It works by adding payment middleware to your endpoints, allowing you to charge users per request.

## Quick Start

### 1. Install Dependencies

```bash
pip install swarms swarms-tools x402 fastapi uvicorn python-dotenv
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key
EXA_API_KEY=your-exa-api-key
```

### 3. Update the Wallet Address

Edit `research_agent_x402_example.py` and replace:

```python
pay_to_address="0xYourWalletAddressHere"
```

with your actual EVM-compatible wallet address (e.g., MetaMask, Coinbase Wallet).

### 4. Run the Server

```bash
python research_agent_x402_example.py
```

The server will start at `http://localhost:8000`

### 5. Test the Integration

#### Free Health Check Endpoint

```bash
curl http://localhost:8000/
```

#### Paid Research Endpoint (requires payment)

```bash
curl http://localhost:8000/research?query="What are the latest breakthroughs in quantum computing?"
```

This will return a 402 Payment Required response with payment instructions.

## How It Works

1. **Agent Creation**: The code creates a Swarms agent with the `exa_search` tool for conducting research
2. **Payment Middleware**: x402's `require_payment` middleware protects the `/research` endpoint
3. **Payment Flow**:
   - Client requests the endpoint
   - Server responds with 402 Payment Required + payment instructions
   - Client makes payment (handled by x402 client SDK)
   - Client retries request with payment proof
   - Server verifies payment and returns research results

## Going to Production (Mainnet)

For testnet (current setup):

- Network: `base-sepolia`
- Uses free facilitator: `https://x402.org/facilitator`
- Test USDC only

For mainnet:

1. Get CDP API credentials from [cdp.coinbase.com](https://cdp.coinbase.com)
2. Update the code:

   ```python
   from cdp.x402 import create_facilitator_config

   facilitator_config = create_facilitator_config(
       api_key_id=os.getenv("CDP_API_KEY_ID"),
       api_key_secret=os.getenv("CDP_API_KEY_SECRET"),
   )

   # Change network to "base" and add facilitator_config
   require_payment(
       path="/research",
       price="$0.01",
       pay_to_address="0xYourWalletAddress",
       network_id="base",  # Changed from base-sepolia
       facilitator_config=facilitator_config,
       ...
   )
   ```

## Spending Limits Example

For per-agent budgets, approval thresholds, and circuit breakers, see:

- `agent_integration/x402_agent_buying.py`

## Learn More

- [X402 Documentation](https://docs.cdp.coinbase.com/x402)
- [Swarms Documentation](https://docs.swarms.world)
- [X402 GitHub Examples](https://github.com/coinbase/x402/tree/main/examples)
