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
                    ? \`✅ Success: \${data.result.analysis}\`
                    : \`❌ Error: \${data.error}\`;
                } catch (e) {
                  document.getElementById('result').innerHTML = \`❌ Failed: \${e.message}\`;
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
    
    console.log(\`✅ Autonomous data collection successful: \${validData.length} sources\`);
    
    // Step 2: Deploy Swarms AI agents for autonomous analysis
    console.log('🧠 Deploying Swarms AI agents for autonomous intelligence generation...');
    const swarmConfiguration = {
      name: "Autonomous Market Intelligence Swarm",
      description: "Self-executing financial analysis and decision support system",
      agents: [
        {
          agent_name: "Autonomous Technical Intelligence Agent",
          system_prompt: \`You are an autonomous technical analysis AI agent operating 24/7. Provide:
            - Real-time trend identification and momentum analysis
            - Dynamic support/resistance level calculations
            - Technical indicator signals (RSI, MACD, moving averages)
            - Autonomous price targets and risk assessments
            - Self-executing trading signal recommendations
            
            Format your analysis as a professional autonomous intelligence briefing for automated systems.\`,
          model_name: "gpt-4o-mini",
          max_tokens: 2500,
          temperature: 0.2
        },
        {
          agent_name: "Autonomous Market Sentiment Agent",
          system_prompt: \`You are an autonomous market sentiment analysis AI agent. Continuously evaluate:
            - Real-time market psychology and investor behavior patterns
            - Volume analysis and institutional activity detection
            - Risk-on vs risk-off sentiment shifts
            - Autonomous sector rotation and leadership identification
            - Self-executing market timing recommendations
            
            Provide actionable intelligence for autonomous decision-making systems.\`,
          model_name: "gpt-4o-mini", 
          max_tokens: 2500,
          temperature: 0.3
        }
      ],
      swarm_type: "ConcurrentWorkflow",
      task: \`Execute autonomous analysis of real-time market intelligence:
      
      LIVE MARKET INTELLIGENCE:
      \${JSON.stringify(marketIntelligence, null, 2)}
      
      Generate comprehensive autonomous intelligence report including:
      1. Technical analysis with specific autonomous entry/exit recommendations
      2. Market sentiment assessment with timing signals for automated systems
      3. Risk management protocols for autonomous execution
      4. Self-executing action recommendations
      5. Key monitoring parameters for next autonomous cycle
      
      Focus on actionable intelligence for autonomous trading systems and automated decision making.\`,
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
      throw new Error(\`Swarms API autonomous execution failed: \${response.status} - \${errorText}\`);
    }

    const analysisResult = await response.json();
    const intelligenceReport = analysisResult.output;
    
    console.log('✅ Autonomous Swarms AI analysis completed successfully');
    console.log(\`💰 Autonomous execution cost: \${analysisResult.usage?.billing_info?.total_cost || 'N/A'}\`);
    
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
        \`https://query1.finance.yahoo.com/v8/finance/chart/\${symbol}\`,
        { 
          signal: controller.signal,
          headers: { 
            'User-Agent': 'Mozilla/5.0 (autonomous-swarms-agent) AppleWebKit/537.36'
          }
        }
      );
      clearTimeout(timeout);
      
      if (!response.ok) {
        throw new Error(\`Autonomous data collection failed: HTTP \${response.status}\`);
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
      
      console.log(\`📈 \${symbol}: $\${currentPrice} (\${changePercent}%) - Autonomous collection successful\`);

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
      console.error(\`❌ Autonomous collection failed for \${symbol}:\`, error.message);
      return [symbol, { 
        error: \`Autonomous collection failed: \${error.message}\`,
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
  console.log(\`📊 Autonomous intelligence collection completed: \${successfulCollections}/\${targetSymbols.length} successful\`);

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
      .map(([symbol, data]) => \`\${symbol}: \${data.change_percent}%\`)
      .join(', ');

    const alertSubject = \`🤖 Autonomous Market Intelligence Alert - \${new Date().toLocaleDateString()}\`;
    
    const alertBody = \`
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; background: #f8f9fa; }
          .container { background: white; margin: 20px; border-radius: 15px; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.1); }
          .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }
          .content { padding: 40px; }
          .autonomous-badge { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 10px; margin: 25px 0; text-align: center; }
          .intelligence-section { background: #f8f9fa; padding: 30px; border-radius: 10px; margin: 25px 0; border-left: 5px solid #667eea; }
          .market-data { margin: 25px 0; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { padding: 15px; text-align: left; border-bottom: 1px solid #eee; }
          th { background: #f8f9fa; font-weight: 600; color: #333; }
          .positive { color: #28a745; font-weight: bold; }
          .negative { color: #dc3545; font-weight: bold; }
          .footer { background: #f8f9fa; padding: 25px; text-align: center; color: #666; font-size: 14px; }
          .badge { display: inline-block; background: #667eea; color: white; padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: 600; margin: 3px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🤖 Autonomous Market Intelligence</h1>
            <p>AI-Powered Autonomous Financial Analysis • Swarms API</p>
            <div>
              <span class="badge">Autonomous</span>
              <span class="badge">Real-time</span>
              <span class="badge">AI-Powered</span>
              <span class="badge">Global Edge</span>
            </div>
            <p><strong>Generated:</strong> \${new Date().toLocaleString()}</p>
          </div>
          
          <div class="content">
            <div class="autonomous-badge">
              <h3>🚀 Autonomous Analysis Execution Complete</h3>
              <p><strong>Significant Market Movements Detected:</strong> \${significantMovements || 'Market within normal volatility parameters'}</p>
              <p><strong>Next Autonomous Cycle:</strong> Scheduled automatically</p>
            </div>
            
            <div class="intelligence-section">
              <h2>🧠 Autonomous AI Intelligence Report</h2>
              <p><em>Generated by autonomous Swarms AI agents with real-time market intelligence:</em></p>
              <pre style="white-space: pre-wrap; font-family: 'Courier New', monospace; background: white; padding: 25px; border-radius: 8px; font-size: 13px; line-height: 1.6; border: 1px solid #eee;">\${intelligenceReport}</pre>
            </div>
            
            <div class="market-data">
              <h2>📊 Real-Time Market Intelligence</h2>
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>Change</th>
                    <th>Volume</th>
                    <th>Market State</th>
                    <th>Data Quality</th>
                  </tr>
                </thead>
                <tbody>
                  \${Object.entries(marketIntelligence)
                    .filter(([symbol, data]) => !data.error)
                    .map(([symbol, data]) => \`
                    <tr>
                      <td><strong>\${symbol}</strong></td>
                      <td>$\${data.price?.toFixed(2) || 'N/A'}</td>
                      <td class="\${parseFloat(data.change_percent) >= 0 ? 'positive' : 'negative'}">
                        \${parseFloat(data.change_percent) >= 0 ? '+' : ''}\${data.change_percent}%
                      </td>
                      <td>\${data.volume?.toLocaleString() || 'N/A'}</td>
                      <td>\${data.market_state || 'N/A'}</td>
                      <td>\${data.data_quality || 'N/A'}</td>
                    </tr>
                  \`).join('')}
                </tbody>
              </table>
            </div>
          </div>
          
          <div class="footer">
            <p><strong>🤖 Powered by Autonomous Swarms AI Agents</strong></p>
            <p>This intelligence report was generated automatically by our autonomous market analysis system</p>
            <p><em>Next autonomous analysis will execute automatically on schedule</em></p>
            <p>Swarms API • Autonomous Intelligence • Edge Computing</p>
          </div>
        </div>
      </body>
      </html>
    \`;

    const formData = new FormData();
    formData.append('from', \`Autonomous Market Intelligence <intelligence@\${env.MAILGUN_DOMAIN}>\`);
    formData.append('to', env.AUTONOMOUS_ALERTS_EMAIL);
    formData.append('subject', alertSubject);
    formData.append('html', alertBody);

    const response = await fetch(\`https://api.mailgun.net/v3/\${env.MAILGUN_DOMAIN}/messages\`, {
      method: 'POST',
      headers: {
        'Authorization': \`Basic \${btoa(\`api:\${env.MAILGUN_API_KEY}\`)}\`
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

## Healthcare Cron Agent Example

Healthcare monitoring cron agent with Swarms AI:

```javascript
export default {
  async fetch(request, env, ctx) {
    if (request.url.includes('/health')) {
      return new Response(JSON.stringify({ 
        status: 'Healthcare cron agent active',
        next_check: 'Every 30 minutes'
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    return new Response('Healthcare Cron Agent');
  },

  // Healthcare monitoring - every 30 minutes
  async scheduled(event, env, ctx) {
    console.log('🏥 Healthcare cron agent triggered');
    ctx.waitUntil(executeHealthAnalysis(event, env));
  }
};

async function executeHealthAnalysis(event, env) {
  try {
    // Collect patient data (from EMR, IoT devices, etc.)
    const patientData = await getPatientData();
    
    // Configure Swarms healthcare agents
    const healthConfig = {
      name: "Healthcare Monitoring Swarm",
      agents: [
        {
          agent_name: "Vital Signs Monitor",
          system_prompt: "Monitor patient vital signs and detect anomalies. Alert on critical values.",
          model_name: "gpt-4o-mini",
          max_tokens: 1000
        }
      ],
      swarm_type: "ConcurrentWorkflow",
      task: `Analyze patient data: ${JSON.stringify(patientData, null, 2)}`,
      max_loops: 1
    };

    // Call Swarms API
    const response = await fetch('https://api.swarms.world/v1/swarm/completions', {
      method: 'POST',
      headers: {
        'x-api-key': env.SWARMS_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(healthConfig)
    });

    const result = await response.json();
    
    // Send alerts if critical conditions detected
    if (result.output.includes('CRITICAL')) {
      await sendHealthAlert(env, result.output);
    }

    return { success: true, analysis: result.output };
  } catch (error) {
    console.error('Healthcare analysis failed:', error);
    return { success: false, error: error.message };
  }
}

async function getPatientData() {
  // Mock patient data - replace with real EMR/IoT integration
  return {
    patient_001: {
      heart_rate: 115, // Elevated
      oxygen_saturation: 89 // Low - critical
    }
  };
}

async function sendHealthAlert(env, analysis) {
  // Send emergency alerts via email/SMS
  console.log('🚨 Critical health alert sent');
}
```

## Deployment & Configuration

### Environment Variables

Configure your Cloudflare Workers deployment with Swarms API:

```jsonc
{
  "vars": {
    "SWARMS_API_KEY": "your-swarms-api-key",
    "AUTONOMOUS_ALERTS_EMAIL": "intelligence@yourcompany.com",
    "HEALTHCARE_EMERGENCY_EMAIL": "emergency@hospital.com",
    "MAILGUN_API_KEY": "your-mailgun-key",
    "MAILGUN_DOMAIN": "intelligence.yourcompany.com"
  }
}
```

### Cloudflare Workers Cron Scheduling Patterns

```jsonc
{
  "triggers": {
    "crons": [
      "0 */3 * * *",        // Financial Swarms agents every 3 hours
      "*/30 * * * *",       // Healthcare Swarms monitoring every 30 minutes  
      "0 9,15,21 * * *",    // Daily Swarms intelligence briefings
      "*/5 * * * *"         // Critical Swarms systems every 5 minutes
    ]
  }
}
```

### Cloudflare Workers Deployment Commands

```bash
# Deploy Swarms AI agents to Cloudflare Workers
wrangler deploy

# Monitor Cloudflare Workers execution logs
wrangler tail

# Test Cloudflare Workers cron triggers manually
wrangler triggers cron "0 */3 * * *"

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