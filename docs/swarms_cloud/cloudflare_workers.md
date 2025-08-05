# Deploy Cron Agents with Swarms API on Cloudflare Workers

Deploy intelligent, self-executing AI agents powered by Swarms API on Cloudflare's global edge network. Build production-ready cron agents that run on automated schedules, fetch real-time data, perform sophisticated analysis using Swarms AI, and take automated actions across 330+ cities worldwide.

## What Are Cron Agents?

Cron agents combine **Swarms API intelligence** with **Cloudflare Workers edge computing** to create AI-powered systems that:

* **Execute automatically** on predefined schedules without human intervention
* **Fetch real-time data** from external sources (APIs, databases, IoT sensors)
* **Perform intelligent analysis** using specialized Swarms AI agents
* **Take automated actions** based on analysis findings (alerts, reports, decisions)
* **Scale globally** on Cloudflare's edge network with sub-100ms response times worldwide

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Cloudflare      │    │  Data Sources   │    │   Swarms API    │
│ Workers Cron    │    │                 │    │                 │
│ "0 */3 * * *"   │───▶│ Yahoo Finance   │───▶│ Multi-Agent     │
│                 │    │ Medical APIs    │    │ Intelligence    │
│ scheduled()     │    │ News Feeds      │    │ Autonomous      │
│ Global Edge     │    │ IoT Sensors     │    │ Actions         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Benefits:**

* **24/7 Operation**: Zero human intervention required
* **Global Edge Deployment**: Cloudflare's 330+ city network for ultra-low latency
* **Swarms AI Intelligence**: Live data analysis with specialized AI agents
* **Automated Decision Making**: Smart actions based on Swarms agent insights
* **Enterprise Reliability**: Production-grade error handling and monitoring

## Quick Start: Deploy Stock Analysis Cron Agent

Create your first financial intelligence cron agent powered by Swarms API and deployed on Cloudflare Workers edge network.

### 1. Cloudflare Workers Project Setup

```bash
# Create new Cloudflare Workers project
npm create cloudflare@latest stock-cron-agent
cd stock-cron-agent

# Install dependencies for Swarms API integration
npm install
```

### 2. Configure Cloudflare Workers Cron Schedule

Edit `wrangler.jsonc` to set up cron execution:

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "stock-cron-agent",
  "main": "src/index.js",
  "compatibility_date": "2025-08-03",
  "observability": {
    "enabled": true
  },
  "triggers": {
    "crons": [
      "0 */3 * * *"  // Cloudflare Workers cron: analysis every 3 hours
    ]
  },
  "vars": {
    "SWARMS_API_KEY": "your-swarms-api-key"  // Your Swarms API key
  }
}
```

### 3. Cloudflare Workers + Swarms API Implementation

Create `src/index.js`:

```javascript
export default {
  // Cloudflare Workers fetch handler - Web interface for monitoring
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    if (url.pathname === '/') {
      return new Response(`
        <html>
          <body>
            <h1>Stock Cron Agent</h1>
            <p>Status: Active | <button onclick="run()">Execute Now</button></p>
            <div id="result"></div>
            <script>
              async function run() {
                document.getElementById('result').innerHTML = 'Running...';
                try {
                  const res = await fetch('/execute');
                  const data = await res.json();
                  document.getElementById('result').innerHTML = data.result?.success 
                    ? '✅ Success: ' + data.result.analysis
                    : '❌ Error: ' + data.error;
                } catch (e) {
                  document.getElementById('result').innerHTML = '❌ Failed: ' + e.message;
                }
              }
            </script>
          </body>
        </html>
      `, {
        headers: { 'Content-Type': 'text/html' }
      });
    }
    
    if (url.pathname === '/execute') {
      try {
        const result = await executeAnalysis(null, env);
        return new Response(JSON.stringify({ 
          message: 'Autonomous analysis executed successfully',
          timestamp: new Date().toISOString(),
          result 
        }), {
          headers: { 'Content-Type': 'application/json' }
        });
      } catch (error) {
        return new Response(JSON.stringify({ 
          error: error.message,
          timestamp: new Date().toISOString(),
          system: 'autonomous-agent'
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }
    
    return new Response('Autonomous Agent Endpoint Not Found', { status: 404 });
  },

  // Cloudflare Workers cron handler - triggered by scheduled events
  async scheduled(event, env, ctx) {
    console.log('🚀 Cloudflare Workers cron triggered - executing Swarms AI analysis');
    ctx.waitUntil(executeAnalysis(event, env));
  }
};

// Core function combining Cloudflare Workers execution with Swarms API intelligence
async function executeAnalysis(event, env) {
  console.log('🤖 Cloudflare Workers executing Swarms AI analysis...');
  
  try {
    // Step 1: Autonomous data collection from multiple sources
    console.log('📊 Executing autonomous data collection...');
    const marketIntelligence = await collectMarketIntelligence();
    
    const validData = Object.keys(marketIntelligence).filter(symbol => !marketIntelligence[symbol].error);
    if (validData.length === 0) {
      throw new Error('Autonomous data collection failed - no valid market intelligence gathered');
    }
    
    console.log('✅ Autonomous data collection successful: ' + validData.length + ' sources');
    
    // Step 2: Deploy Swarms AI agents for autonomous analysis
    console.log('🧠 Deploying Swarms AI agents for autonomous intelligence generation...');
    const swarmConfiguration = {
      name: "Autonomous Market Intelligence Swarm",
      description: "Self-executing financial analysis and decision support system",
      agents: [
        {
          agent_name: "Autonomous Technical Intelligence Agent",
          system_prompt: "You are an autonomous technical analysis AI agent operating 24/7. Provide: Real-time trend identification and momentum analysis, dynamic support/resistance level calculations, technical indicator signals (RSI, MACD, moving averages), autonomous price targets and risk assessments, and self-executing trading signal recommendations. Format your analysis as a professional autonomous intelligence briefing for automated systems.",
          model_name: "gpt-4o-mini",
          max_tokens: 2500,
          temperature: 0.2
        },
        {
          agent_name: "Autonomous Market Sentiment Agent",
          system_prompt: "You are an autonomous market sentiment analysis AI agent. Continuously evaluate: Real-time market psychology and investor behavior patterns, volume analysis and institutional activity detection, risk-on vs risk-off sentiment shifts, autonomous sector rotation and leadership identification, and self-executing market timing recommendations. Provide actionable intelligence for autonomous decision-making systems.",
          model_name: "gpt-4o-mini", 
          max_tokens: 2500,
          temperature: 0.3
        }
      ],
      swarm_type: "ConcurrentWorkflow",
      task: `Execute autonomous analysis of real-time market intelligence: ${JSON.stringify(marketIntelligence, null, 2)} Generate comprehensive autonomous intelligence report including: 1. Technical analysis with specific autonomous entry/exit recommendations 2. Market sentiment assessment with timing signals for automated systems 3. Risk management protocols for autonomous execution 4. Self-executing action recommendations 5. Key monitoring parameters for next autonomous cycle Focus on actionable intelligence for autonomous trading systems and automated decision making.`,
      max_loops: 1
    };

    // Execute Swarms API call from Cloudflare Workers edge
    const response = await fetch('https://api.swarms.world/v1/swarm/completions', {
      method: 'POST',
      headers: {
        'x-api-key': env.SWARMS_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(swarmConfiguration)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error('Swarms API autonomous execution failed: ' + response.status + ' - ' + errorText);
    }

    const analysisResult = await response.json();
    const intelligenceReport = analysisResult.output;
    
    console.log('✅ Autonomous Swarms AI analysis completed successfully');
    console.log('💰 Autonomous execution cost: ' + (analysisResult.usage?.billing_info?.total_cost || 'N/A'));
    
    // Step 3: Execute autonomous actions based on intelligence
    if (env.AUTONOMOUS_ALERTS_EMAIL) {
      console.log('📧 Executing autonomous alert system...');
      await executeAutonomousAlerts(env, intelligenceReport, marketIntelligence);
    }

    return {
      success: true,
      analysis: intelligenceReport,
      symbolsAnalyzed: validData.length,
      cost: analysisResult.usage?.billing_info?.total_cost || analysisResult.metadata?.billing_info?.total_cost,
      executionTime: new Date().toISOString(),
      nextExecution: 'Scheduled for next autonomous cron trigger',
      autonomousSystem: 'Swarms AI Agents'
    };

  } catch (error) {
    console.error('❌ Autonomous analysis execution failed:', error.message);
    return {
      success: false,
      error: error.message,
      executionTime: new Date().toISOString(),
      autonomousSystem: 'Error in autonomous pipeline'
    };
  }
}

// Autonomous market intelligence collection
async function collectMarketIntelligence() {
  const targetSymbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA'];
  const marketIntelligence = {};

  console.log('🎯 Executing autonomous multi-source data collection...');

  const dataCollectionPromises = targetSymbols.map(async (symbol) => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      
      // Autonomous data collection from Yahoo Finance API
      const response = await fetch(
        'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol,
        { 
          signal: controller.signal,
          headers: { 
            'User-Agent': 'Mozilla/5.0 (autonomous-swarms-agent) AppleWebKit/537.36'
          }
        }
      );
      clearTimeout(timeout);
      
      if (!response.ok) {
        throw new Error('Autonomous data collection failed: HTTP ' + response.status);
      }
      
      const data = await response.json();
      const chartResult = data.chart.result[0];
      const meta = chartResult.meta;
      
      if (!meta) {
        throw new Error('Invalid market intelligence structure received');
      }
      
      const currentPrice = meta.regularMarketPrice;
      const previousClose = meta.previousClose;
      const dayChange = currentPrice - previousClose;
      const changePercent = ((dayChange / previousClose) * 100).toFixed(2);
      
      console.log('📈 ' + symbol + ': $' + currentPrice + ' (' + changePercent + '%) - Autonomous collection successful');

      return [symbol, {
        price: currentPrice,
        change: dayChange,
        change_percent: changePercent,
        volume: meta.regularMarketVolume || 0,
        market_cap: meta.marketCap || 0,
        pe_ratio: meta.trailingPE || 0,
        day_high: meta.regularMarketDayHigh,
        day_low: meta.regularMarketDayLow,
        fifty_two_week_high: meta.fiftyTwoWeekHigh,
        fifty_two_week_low: meta.fiftyTwoWeekLow,
        currency: meta.currency || 'USD',
        market_state: meta.marketState,
        autonomous_collection_time: new Date().toISOString(),
        data_quality: 'high'
      }];

    } catch (error) {
      console.error('❌ Autonomous collection failed for ' + symbol + ':', error.message);
      return [symbol, { 
        error: 'Autonomous collection failed: ' + error.message,
        autonomous_collection_time: new Date().toISOString(),
        data_quality: 'failed'
      }];
    }
  });

  const results = await Promise.allSettled(dataCollectionPromises);
  results.forEach((result) => {
    if (result.status === 'fulfilled' && result.value) {
      const [symbol, data] = result.value;
      marketIntelligence[symbol] = data;
    }
  });

  const successfulCollections = Object.keys(marketIntelligence).filter(k => !marketIntelligence[k]?.error).length;
  console.log('📊 Autonomous intelligence collection completed: ' + successfulCollections + '/' + targetSymbols.length + ' successful');

  return marketIntelligence;
}

// Autonomous alert and notification system
async function executeAutonomousAlerts(env, intelligenceReport, marketIntelligence) {
  if (!env.MAILGUN_API_KEY || !env.MAILGUN_DOMAIN || !env.AUTONOMOUS_ALERTS_EMAIL) {
    console.log('⚠️ Autonomous alert system not configured - skipping notifications');
    return;
  }

  try {
    // Autonomous detection of significant market movements
    const significantMovements = Object.entries(marketIntelligence)
      .filter(([symbol, data]) => data.change_percent && Math.abs(parseFloat(data.change_percent)) > 3)
      .map(([symbol, data]) => symbol + ': ' + data.change_percent + '%')
      .join(', ');

    const alertSubject = '🤖 Autonomous Market Intelligence Alert - ' + new Date().toLocaleDateString();
    
    const alertBody = `
      <html>
        <body>
          <h1>🤖 Market Intelligence Alert</h1>
          <p><strong>Date:</strong> ${new Date().toLocaleString()}</p>
          <p><strong>Movements:</strong> ${significantMovements || 'Normal volatility'}</p>
          
          <h2>Analysis Report:</h2>
          <pre>${intelligenceReport}</pre>
          
          <p>Powered by Swarms API</p>
        </body>
      </html>
    `;

    const formData = new FormData();
    formData.append('from', 'Autonomous Market Intelligence <intelligence@' + env.MAILGUN_DOMAIN + '>');
    formData.append('to', env.AUTONOMOUS_ALERTS_EMAIL);
    formData.append('subject', alertSubject);
    formData.append('html', alertBody);

    const response = await fetch('https://api.mailgun.net/v3/' + env.MAILGUN_DOMAIN + '/messages', {
      method: 'POST',
      headers: {
        'Authorization': 'Basic ' + btoa('api:' + env.MAILGUN_API_KEY)
      },
      body: formData
    });

    if (response.ok) {
      console.log('✅ Autonomous alert system executed successfully');
    } else {
      console.error('❌ Autonomous alert system execution failed:', await response.text());
    }
    
  } catch (error) {
    console.error('❌ Autonomous alert system error:', error.message);
  }
}
```

## Production Best Practices

### 1. **Cloudflare Workers + Swarms API Integration**

* Implement comprehensive error handling for both platforms
* Use Cloudflare Workers KV for caching Swarms API responses
* Leverage Cloudflare Workers analytics for monitoring

### 2. **Cost Optimization** 

* Monitor Swarms API usage and costs
* Use Cloudflare Workers free tier (100K requests/day)
* Implement intelligent batching for Swarms API efficiency
* Use cost-effective Swarms models (gpt-4o-mini recommended)

### 3. **Security & Compliance**

* Secure Swarms API keys in Cloudflare Workers environment variables
* Use Cloudflare Workers secrets for sensitive data
* Audit AI decisions and maintain compliance logs
* HIPAA compliance for healthcare applications

### 4. **Monitoring & Observability**

* Track Cloudflare Workers performance metrics
* Monitor Swarms API response times and success rates
* Use Cloudflare Workers analytics dashboard
* Set up alerts for system failures and anomalies

This deployment architecture combines **Swarms API's advanced multi-agent intelligence** with **Cloudflare Workers' global edge infrastructure**, enabling truly intelligent, self-executing AI agents that operate continuously across 330+ cities worldwide, providing real-time intelligence and automated decision-making capabilities with ultra-low latency.