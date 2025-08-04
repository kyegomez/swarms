# Cloudflare Workers with Swarms: Production AI Agents

Deploy AI agents on Cloudflare's edge network with automatic cron scheduling and real-time data integration. This guide shows a production-ready implementation with both HTTP endpoints and scheduled triggers.

## Architecture Overview

The Cloudflare Workers pattern uses two main handlers:
- **`fetch()`**: HTTP requests for testing and manual triggers
- **`scheduled()`**: Cron jobs for automated execution

```javascript
export default {
  // HTTP handler for manual testing
  async fetch(request, env, ctx) {
    // Handle web interface and API endpoints
  },
  
  // Cron handler for scheduled execution
  async scheduled(event, env, ctx) {
    ctx.waitUntil(handleStockAnalysis(event, env));
  }
};
```

## Quick Setup

### 1. Create Worker Project

```bash
npm create cloudflare@latest stock-agent
cd stock-agent
```

### 2. Configure Cron Schedule

Edit `wrangler.jsonc`:

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "stock-agent",
  "main": "src/index.js",
  "compatibility_date": "2025-08-03",
  "observability": {
    "enabled": true
  },
  "triggers": {
    "crons": [
      "0 */3 * * *"  // Every 3 hours
    ]
  },
  "vars": {
    "SWARMS_API_KEY": "your-api-key"
  }
}
```

## Complete Minimal Implementation

Create `src/index.js`:

```javascript
export default {
  // HTTP handler - provides web interface and manual triggers
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Web interface for testing
    if (url.pathname === '/') {
      return new Response(`
        <!DOCTYPE html>
        <html>
          <head>
            <title>Stock Analysis Agent</title>
            <style>
              body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
              .btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
              .status { background: #f0f8f0; padding: 10px; border-left: 3px solid #28a745; margin: 20px 0; }
              .result { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; white-space: pre-wrap; }
            </style>
          </head>
          <body>
            <h1>📈 Stock Analysis Agent</h1>
            <div class="status">Status: Online ✅</div>
            
            <button class="btn" onclick="triggerAnalysis()">🔥 Start Analysis</button>
            <div id="result"></div>
            
            <script>
              async function triggerAnalysis() {
                document.getElementById('result').innerHTML = 'Running analysis...';
                try {
                  const response = await fetch('/trigger');
                  const data = await response.json();
                  document.getElementById('result').innerHTML = 
                    data.result?.success ? 
                    \`✅ Success\\n\\nAnalysis: \\${data.result.analysis}\` :
                    \`❌ Error: \\${data.result?.error || data.error}\`;
                } catch (error) {
                  document.getElementById('result').innerHTML = \`❌ Failed: \\${error.message}\`;
                }
              }
            </script>
          </body>
        </html>
      `, {
        headers: { 'Content-Type': 'text/html' }
      });
    }
    
    // Manual trigger endpoint
    if (url.pathname === '/trigger') {
      try {
        const result = await handleStockAnalysis(null, env);
        return new Response(JSON.stringify({ 
          message: 'Analysis triggered',
          timestamp: new Date().toISOString(),
          result 
        }), {
          headers: { 'Content-Type': 'application/json' }
        });
      } catch (error) {
        return new Response(JSON.stringify({ 
          error: error.message 
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }
    
    return new Response('Not Found', { status: 404 });
  },

  // Cron handler - runs automatically on schedule
  async scheduled(event, env, ctx) {
    ctx.waitUntil(handleStockAnalysis(event, env));
  }
};

// Main analysis function used by both HTTP and cron triggers
async function handleStockAnalysis(event, env) {
  console.log('🚀 Starting stock analysis...');
  
  try {
    // Step 1: Fetch real market data (using Yahoo Finance - no API key needed)
    const marketData = await fetchMarketData();
    
    // Check if we got valid data
    const validSymbols = Object.keys(marketData).filter(symbol => !marketData[symbol].error);
    if (validSymbols.length === 0) {
      throw new Error('No valid market data retrieved');
    }
    
    // Step 2: Send to Swarms AI agents
    const swarmConfig = {
      name: "Stock Analysis",
      description: "Real-time market analysis",
      agents: [
        {
          agent_name: "Technical Analyst",
          system_prompt: \`Analyze the provided stock data:
            - Identify trends and key levels
            - Provide trading signals
            - Calculate technical indicators
            Format analysis professionally.\`,
          model_name: "gpt-4o-mini",
          max_tokens: 1500,
          temperature: 0.2
        }
      ],
      swarm_type: "SequentialWorkflow",
      task: \`Analyze this market data: \\${JSON.stringify(marketData, null, 2)}\`,
      max_loops: 1
    };

    const response = await fetch('https://api.swarms.world/v1/swarm/completions', {
      method: 'POST',
      headers: {
        'x-api-key': env.SWARMS_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(swarmConfig)
    });

    if (!response.ok) {
      throw new Error(\`Swarms API error: \\${response.status}\`);
    }

    const result = await response.json();
    console.log('✅ Analysis completed');

    return {
      success: true,
      analysis: result.output,
      symbolsAnalyzed: validSymbols.length,
      cost: result.usage?.billing_info?.total_cost
    };

  } catch (error) {
    console.error('❌ Analysis failed:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

// Fetch market data from Yahoo Finance (free, no API key required)
async function fetchMarketData() {
  const symbols = ['SPY', 'AAPL', 'MSFT', 'TSLA'];
  const marketData = {};

  const promises = symbols.map(async (symbol) => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      
      const response = await fetch(
        \`https://query1.finance.yahoo.com/v8/finance/chart/\\${symbol}\`,
        { 
          signal: controller.signal,
          headers: { 'User-Agent': 'Mozilla/5.0' }
        }
      );
      clearTimeout(timeout);
      
      if (!response.ok) throw new Error(\`HTTP \\${response.status}\`);
      
      const data = await response.json();
      const result = data.chart.result[0];
      const meta = result.meta;
      
      const currentPrice = meta.regularMarketPrice;
      const previousClose = meta.previousClose;
      const change = currentPrice - previousClose;
      const changePercent = ((change / previousClose) * 100).toFixed(2);

      return [symbol, {
        price: currentPrice,
        change: change,
        change_percent: changePercent,
        volume: meta.regularMarketVolume,
        currency: meta.currency
      }];

    } catch (error) {
      return [symbol, { error: error.message }];
    }
  });

  const results = await Promise.allSettled(promises);
  results.forEach((result) => {
    if (result.status === 'fulfilled' && result.value) {
      const [symbol, data] = result.value;
      marketData[symbol] = data;
    }
  });

  return marketData;
}
```

## Key Features Explained

### 1. **Dual Handler Pattern**
- **`fetch()`**: Handles HTTP requests, provides web UI for testing
- **`scheduled()`**: Executes on cron schedule automatically
- **Shared Logic**: Both use the same `handleStockAnalysis()` function

### 2. **Cron Configuration**
```jsonc
"triggers": {
  "crons": [
    "0 */3 * * *"    // Every 3 hours
    "0 9 * * MON-FRI" // Weekdays at 9 AM
  ]
}
```

### 3. **Real Data Integration**
- **Yahoo Finance API**: Free, no API key required
- **Error Handling**: Timeout management, fallback responses
- **Parallel Processing**: Fetch multiple symbols simultaneously

### 4. **Production Features**
- **Web Interface**: Test manually via browser
- **Structured Responses**: Consistent JSON format
- **Error Recovery**: Graceful failure handling
- **Logging**: Console output for debugging

## Deployment

```bash
# Deploy to Cloudflare
wrangler deploy

# View logs
wrangler tail

# Test cron manually
wrangler triggers cron "0 */3 * * *"
```

## Environment Variables

Add to `wrangler.jsonc`:

```jsonc
{
  "vars": {
    "SWARMS_API_KEY": "your-swarms-api-key",
    "MAILGUN_API_KEY": "optional-for-emails", 
    "MAILGUN_DOMAIN": "your-domain.com",
    "RECIPIENT_EMAIL": "alerts@company.com"
  }
}
```

## Testing

1. **Deploy**: `wrangler deploy`
2. **Visit URL**: Open your worker URL to see the web interface
3. **Manual Test**: Click "Start Analysis" button
4. **Cron Test**: `wrangler triggers cron "0 */3 * * *"`

## Production Tips

- **Error Handling**: Always wrap API calls in try-catch
- **Timeouts**: Use AbortController for external API calls
- **Logging**: Use console.log for debugging in Cloudflare dashboard
- **Rate Limits**: Yahoo Finance is free but has rate limits
- **Cost Control**: Set appropriate `max_tokens` in agent config

This minimal implementation provides a solid foundation for production AI agents on Cloudflare Workers with automated scheduling and real-time data integration.