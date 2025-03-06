# Swarm Agent API Pricing Documentation


The Swarm Agent API provides a powerful platform for managing and executing Python agents in the cloud without requiring Docker or Kubernetes. This document outlines the pricing model, how costs are calculated, and how to purchase and manage your credits.

Our pricing is designed to be transparent and cost-effective, with costs based on:

- Number of agents used

- Input and output token usage 

- Execution time

## Credit System

The Swarm API operates on a credit-based system:

- **Credits** are the currency used within the platform

- 1 credit = $1 USD

- Credits can be purchased with USD or $swarms Solana tokens

- Two types of credits:

  - **Standard Credits**: Purchased credits that never expire

  - **Free Credits**: Promotional credits that may have usage restrictions

## Pricing Structure

### Base Costs

| Cost Component | Price |
|----------------|-------|
| Base cost per agent | $0.01 per agent |

### Token Usage Costs

| Token Type | Cost |
|------------|------|
| Input tokens | $2.00 per 1M tokens |
| Output tokens | $4.50 per 1M tokens |

### Night-Time Discount

To encourage efficient resource usage during off-peak hours, we offer significant discounts for operations performed during California night-time hours:

| Time Period (Pacific Time) | Discount |
|----------------------------|----------|
| 8:00 PM to 6:00 AM | 75% off token costs |

## Cost Calculation

### Formula

The total cost for a swarm execution is calculated as follows:

```
Total Cost = (Number of Agents × $0.01) + 
             (Total Input Tokens / 1M × $2.00 × Number of Agents) +
             (Total Output Tokens / 1M × $4.50 × Number of Agents)
```

With night-time discount applied:
```
Input Token Cost = Input Token Cost × 0.25
Output Token Cost = Output Token Cost × 0.25
```

### Example Scenarios

#### Scenario 1: Basic Workflow (Day-time)
- 3 agents

- 10,000 input tokens total

- 25,000 output tokens total

**Calculation:**
- Agent cost: 3 × $0.01 = $0.03

- Input token cost: (10,000 / 1,000,000) × $2.00 × 3 = $0.06

- Output token cost: (25,000 / 1,000,000) × $4.50 × 3 = $0.3375

- **Total cost: $0.4275**

#### Scenario 2: Complex Workflow (Night-time)

- 5 agents

- 50,000 input tokens total

- 125,000 output tokens total

**Calculation:**
- Agent cost: 5 × $0.01 = $0.05

- Input token cost: (50,000 / 1,000,000) × $2.00 × 5 × 0.25 = $0.125

- Output token cost: (125,000 / 1,000,000) × $4.50 × 5 × 0.25 = $0.703125

- **Total cost: $0.878125**

## Purchasing Credits

Credits can be purchased through our platform in two ways:

1. **USD Payment**
   - Available through our [account page](https://swarms.world/platform/account)
   - Secure payment processing
   - Minimum purchase: $10

2. **$swarms Token Payment**
   - Use Solana-based $swarms tokens
   - Tokens can be purchased on supported exchanges
   - Connect your Solana wallet on our [account page](https://swarms.world/platform/account)

## Free Credits

We occasionally offer free credits to:

- New users (welcome bonus)

- During promotional periods

- For educational and research purposes

Notes about free credits:
- Used before standard credits

- May have expiration dates

- May have usage restrictions

## Billing and Usage Tracking

Track your credit usage through our comprehensive logging and reporting features:

1. **API Logs**
   - Access detailed logs via the `/v1/swarm/logs` endpoint

   - View cost breakdowns for each execution

2. **Dashboard**

   - Real-time credit balance display

   - Historical usage graphs

   - Detailed cost analysis

   - Available at [https://swarms.world/platform/dashboard](https://swarms.world/platform/dashboard)

## FAQ

**Q: Is there a minimum credit purchase?**  
A: Yes, the minimum credit purchase is $10 USD equivalent.

**Q: Do credits expire?**  
A: Standard credits do not expire. Free promotional credits may have expiration dates.

**Q: How is the night-time discount applied?**  
A: The system automatically detects the execution time based on Pacific Time (America/Los_Angeles) and applies a 75% discount to token costs for executions between 8:00 PM and 6:00 AM.

**Q: What happens if I run out of credits during execution?**  
A: Executions will fail with a 402 Payment Required error if sufficient credits are not available. We recommend maintaining a credit balance appropriate for your usage patterns.

**Q: Can I get a refund for unused credits?**  
A: Please contact our support team for refund requests for unused credits.

**Q: Are there volume discounts available?**  
A: Yes, please contact our sales team for enterprise pricing and volume discounts.

## References

- [Swarm API Documentation](https://docs.swarms.world)

- [Account Management Portal](https://swarms.world/platform/account)

- [Swarm Types Reference](https://docs.swarms.world/swarm-types)

- [Token Usage Guide](https://docs.swarms.world/token-usage)

- [API Reference](https://docs.swarms.world/api-reference)

---

For additional questions or custom pricing options, please contact our support team at kye@swarms.world.