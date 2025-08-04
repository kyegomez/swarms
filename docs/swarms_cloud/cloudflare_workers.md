# Cloudflare Workers with Swarms: Automated Cron Job Agents

Deploy scheduled AI agents on Cloudflare's global edge network for automated stock analysis and healthcare monitoring. This guide focuses on cron job agents that run automatically at scheduled intervals.

## Overview

Cloudflare Workers with cron triggers enable automated agent execution for:
- **Stock Market Analysis**: Daily market reports and trading insights
- **Healthcare Monitoring**: Patient data analysis and alerts
- **Automated Reporting**: Scheduled business intelligence
- **Global Deployment**: Edge computing with zero cold starts

## Quick Setup

### 1. Create Worker Project

```bash
npx create-cloudflare stock-agent worker
cd stock-agent
```

### 2. Configure Cron Schedule

Edit `wrangler.toml`:

```toml
name = "stock-analysis-agent"
main = "src/index.js"
compatibility_date = "2024-01-01"

[env.production.vars]
SWARMS_API_KEY = "your-api-key-here"
SLACK_WEBHOOK_URL = "optional-slack-webhook"

# Stock market analysis - after market close
[[env.production.triggers.crons]]
cron = "0 21 * * MON-FRI"  # 9 PM UTC (4 PM EST)

# Healthcare monitoring - every 4 hours
[[env.production.triggers.crons]]
cron = "0 */4 * * *"
```

## Stock Market Analysis Agent

Automated daily stock analysis after market close:

```javascript
export default {
  // Cron job handler - runs automatically
  async scheduled(event, env, ctx) {
    ctx.waitUntil(handleStockAnalysis(event, env));
  }
};

async function handleStockAnalysis(event, env) {
  try {
    // Step 1: Fetch real market data from multiple sources
    const marketData = await fetchMarketData(env);
    
    // Step 2: Get market news
    const marketNews = await fetchMarketNews(env);
    
    // Step 3: Send real data to Swarms agents for analysis
    const swarmConfig = {
      name: "Real-Time Stock Analysis",
      description: "Live market data analysis with AI agents",
      agents: [
        {
          agent_name: "Technical Analyst",
          system_prompt: `You are a professional technical analyst. Analyze the provided real market data:
            - Calculate key technical indicators (RSI, MACD, Moving Averages)
            - Identify support and resistance levels
            - Determine market trends and momentum
            - Provide trading signals and price targets
            Format your analysis professionally with specific price levels.`,
          model_name: "gpt-4o-mini",
          max_tokens: 3000,
          temperature: 0.2
        },
        {
          agent_name: "Fundamental Analyst",
          system_prompt: `You are a fundamental market analyst. Using the provided market news and data:
            - Analyze earnings impact and company fundamentals
            - Evaluate economic indicators and Fed policy effects
            - Assess sector rotation and market sentiment
            - Identify value opportunities and risks
            Provide investment recommendations with risk assessment.`,
          model_name: "gpt-4o-mini",
          max_tokens: 3000,
          temperature: 0.3
        }
      ],
      swarm_type: "ConcurrentWorkflow",
      task: `Analyze the following real market data and news:

MARKET DATA:
${JSON.stringify(marketData, null, 2)}

MARKET NEWS:
${marketNews}

Provide comprehensive analysis with:
1. Technical analysis with key levels
2. Fundamental analysis with catalysts
3. Trading recommendations
4. Risk assessment
5. Tomorrow's key levels to watch`,
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

    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ Real-time stock analysis completed');
      console.log('💰 Cost:', result.metadata?.billing_info?.total_cost);
      
      // Send email directly
      await sendEmailReport(env, result.output, marketData);
    }
  } catch (error) {
    console.error('❌ Real-time stock analysis failed:', error);
  }
}

// Fetch real market data from Alpha Vantage API
async function fetchMarketData(env) {
  const symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA'];
  const marketData = {};

  for (const symbol of symbols) {
    try {
      // Get daily prices
      const priceResponse = await fetch(
        `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&apikey=${env.STOCK_API_KEY}`
      );
      const priceData = await priceResponse.json();
      
      // Get technical indicators (RSI)
      const rsiResponse = await fetch(
        `https://www.alphavantage.co/query?function=RSI&symbol=${symbol}&interval=daily&time_period=14&series_type=close&apikey=${env.STOCK_API_KEY}`
      );
      const rsiData = await rsiResponse.json();

      if (priceData['Time Series (Daily)'] && rsiData['Technical Analysis: RSI']) {
        const latestDate = Object.keys(priceData['Time Series (Daily)'])[0];
        const latestPrice = priceData['Time Series (Daily)'][latestDate];
        const latestRSI = Object.values(rsiData['Technical Analysis: RSI'])[0];

        marketData[symbol] = {
          price: parseFloat(latestPrice['4. close']),
          open: parseFloat(latestPrice['1. open']),
          high: parseFloat(latestPrice['2. high']),
          low: parseFloat(latestPrice['3. low']),
          volume: parseInt(latestPrice['5. volume']),
          change: parseFloat(latestPrice['4. close']) - parseFloat(latestPrice['1. open']),
          change_percent: ((parseFloat(latestPrice['4. close']) - parseFloat(latestPrice['1. open'])) / parseFloat(latestPrice['1. open']) * 100).toFixed(2),
          rsi: parseFloat(latestRSI?.RSI || 50),
          date: latestDate
        };
      }
      
      // Rate limiting - Alpha Vantage allows 5 requests per minute on free tier
      await new Promise(resolve => setTimeout(resolve, 12000));
      
    } catch (error) {
      console.error(`Error fetching data for ${symbol}:`, error);
      marketData[symbol] = { error: 'Failed to fetch data' };
    }
  }

  return marketData;
}

// Fetch market news from Financial Modeling Prep (free tier available)
async function fetchMarketNews(env) {
  try {
    const newsResponse = await fetch(
      `https://financialmodelingprep.com/api/v3/stock_news?tickers=AAPL,MSFT,TSLA,NVDA&limit=10&apikey=${env.FMP_API_KEY || 'demo'}`
    );
    const newsData = await newsResponse.json();
    
    if (Array.isArray(newsData)) {
      return newsData.slice(0, 5).map(article => ({
        title: article.title,
        text: article.text?.substring(0, 300) + '...',
        publishedDate: article.publishedDate,
        symbol: article.symbol,
        url: article.url
      }));
    }
  } catch (error) {
    console.error('Error fetching news:', error);
  }
  
  return "Market news temporarily unavailable";
}

// Send email report using Mailgun API
async function sendEmailReport(env, analysis, marketData) {
  // Extract key market movers for email subject
  const movers = Object.entries(marketData)
    .filter(([symbol, data]) => data.change_percent && Math.abs(parseFloat(data.change_percent)) > 2)
    .map(([symbol, data]) => `${symbol}: ${data.change_percent}%`)
    .join(', ');

  const emailSubject = `📊 Daily Stock Analysis - ${new Date().toLocaleDateString()}`;
  const emailBody = `
    <h2>Daily Market Analysis Report</h2>
    <p><strong>Date:</strong> ${new Date().toLocaleString()}</p>
    <p><strong>Key Market Movers:</strong> ${movers || 'Market stable'}</p>
    
    <h3>AI Agent Analysis:</h3>
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 5px;">
      <pre style="white-space: pre-wrap;">${analysis}</pre>
    </div>
    
    <h3>Market Data Summary:</h3>
    <table border="1" style="border-collapse: collapse; width: 100%;">
      <tr style="background-color: #e0e0e0;">
        <th>Symbol</th><th>Price</th><th>Change %</th><th>Volume</th><th>RSI</th>
      </tr>
      ${Object.entries(marketData).map(([symbol, data]) => `
        <tr>
          <td>${symbol}</td>
          <td>$${data.price?.toFixed(2) || 'N/A'}</td>
          <td style="color: ${parseFloat(data.change_percent) >= 0 ? 'green' : 'red'}">
            ${data.change_percent}%
          </td>
          <td>${data.volume?.toLocaleString() || 'N/A'}</td>
          <td>${data.rsi?.toFixed(1) || 'N/A'}</td>
        </tr>
      `).join('')}
    </table>
    
    <p><em>Generated by Swarms AI Agent System</em></p>
  `;

  // Send via Mailgun
  const formData = new FormData();
  formData.append('from', `Stock Analysis Agent <noreply@${env.MAILGUN_DOMAIN}>`);
  formData.append('to', env.RECIPIENT_EMAIL);
  formData.append('subject', emailSubject);
  formData.append('html', emailBody);

  try {
    const response = await fetch(`https://api.mailgun.net/v3/${env.MAILGUN_DOMAIN}/messages`, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${btoa(`api:${env.MAILGUN_API_KEY}`)}`
      },
      body: formData
    });

    if (response.ok) {
      console.log('✅ Email report sent successfully');
    } else {
      console.error('❌ Failed to send email:', await response.text());
    }
  } catch (error) {
    console.error('❌ Email sending error:', error);
  }
}

async function sendSlackNotification(webhookUrl, analysis) {
  await fetch(webhookUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: "📈 Daily Stock Market Analysis",
      blocks: [{
        type: "section",
        text: {
          type: "mrkdwn",
          text: `*Market Analysis Complete*\n\`\`\`${analysis.substring(0, 500)}...\`\`\``
        }
      }]
    })
  });
}
```

## Healthcare Monitoring Agent

Automated patient monitoring and health alerts:

```javascript
export default {
  async scheduled(event, env, ctx) {
    // Determine which agent to run based on cron schedule
    const hour = new Date().getHours();
    
    if (hour % 4 === 0) {
      // Every 4 hours - patient monitoring
      ctx.waitUntil(handlePatientMonitoring(event, env));
    }
  }
};

async function handlePatientMonitoring(event, env) {
  const healthcareSwarm = {
    name: "Patient Monitoring System",
    description: "Automated patient health analysis and alerting",
    agents: [
      {
        agent_name: "Vital Signs Analyst",
        system_prompt: `Analyze patient vital signs data for abnormalities:
          - Heart rate patterns and irregularities
          - Blood pressure trends
          - Oxygen saturation levels
          - Temperature variations
          Flag critical conditions requiring immediate attention.`,
        model_name: "gpt-4o-mini",
        max_tokens: 2000,
        temperature: 0.1
      },
      {
        agent_name: "Risk Assessment Specialist",
        system_prompt: `Evaluate patient risk factors and health trends:
          - Medication interactions
          - Chronic condition management
          - Recovery progress assessment
          - Early warning signs detection
          Prioritize patients needing urgent care.`,
        model_name: "gpt-4o-mini", 
        max_tokens: 2000,
        temperature: 0.2
      }
    ],
    swarm_type: "ConcurrentWorkflow",
    task: "Analyze current patient monitoring data and generate health status alerts for medical staff.",
    max_loops: 1
  };

  try {
    const response = await fetch('https://api.swarms.world/v1/swarm/completions', {
      method: 'POST',
      headers: {
        'x-api-key': env.SWARMS_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(healthcareSwarm)
    });

    const result = await response.json();
    
    if (response.ok) {
      console.log('🏥 Health monitoring completed');
      
      // Send email alerts for all monitoring results
      await sendHealthEmailAlert(env, result.output);
    }
  } catch (error) {
    console.error('❌ Health monitoring failed:', error);
  }
}

// Send healthcare email alerts
async function sendHealthEmailAlert(env, analysis) {
  const severity = extractSeverity(analysis);
  const isUrgent = severity === 'critical' || severity === 'urgent';
  
  const emailSubject = `${isUrgent ? '🚨 URGENT' : '🏥'} Health Monitoring Alert - ${new Date().toLocaleString()}`;
  const emailBody = `
    <h2 style="color: ${isUrgent ? 'red' : 'blue'};">Patient Monitoring Report</h2>
    <p><strong>Timestamp:</strong> ${new Date().toLocaleString()}</p>
    <p><strong>Severity Level:</strong> <span style="color: ${getSeverityColor(severity)}; font-weight: bold;">${severity.toUpperCase()}</span></p>
    
    <h3>AI Health Analysis:</h3>
    <div style="background-color: ${isUrgent ? '#ffe6e6' : '#f0f8ff'}; padding: 20px; border-radius: 5px; border-left: 5px solid ${isUrgent ? 'red' : 'blue'};">
      <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">${analysis}</pre>
    </div>
    
    ${isUrgent ? '<p style="color: red; font-weight: bold;">⚠️ IMMEDIATE ATTENTION REQUIRED</p>' : ''}
    
    <p><em>Generated by Swarms Healthcare Monitoring Agent</em></p>
  `;

  // Send email using Mailgun
  const formData = new FormData();
  formData.append('from', `Healthcare Monitor <alerts@${env.MAILGUN_DOMAIN}>`);
  formData.append('to', env.MEDICAL_TEAM_EMAIL);
  formData.append('subject', emailSubject);
  formData.append('html', emailBody);

  try {
    const response = await fetch(`https://api.mailgun.net/v3/${env.MAILGUN_DOMAIN}/messages`, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${btoa(`api:${env.MAILGUN_API_KEY}`)}`
      },
      body: formData
    });

    if (response.ok) {
      console.log('✅ Healthcare email alert sent successfully');
    } else {
      console.error('❌ Failed to send healthcare email:', await response.text());
    }
  } catch (error) {
    console.error('❌ Healthcare email error:', error);
  }
}

function extractSeverity(analysis) {
  if (analysis.includes('CRITICAL')) return 'critical';
  if (analysis.includes('URGENT')) return 'urgent';
  if (analysis.includes('WARNING')) return 'warning';
  return 'normal';
}

function getSeverityColor(severity) {
  switch(severity) {
    case 'critical': return 'red';
    case 'urgent': return 'orange';
    case 'warning': return 'yellow';
    default: return 'green';
  }
}
```

## Deployment

```bash
# Deploy your cron job agents
wrangler deploy

# Monitor logs
wrangler tail

# Test cron trigger manually
wrangler triggers cron "0 21 * * MON-FRI"
```

## Cost Optimization

- Use `gpt-4o-mini` for cost-effective analysis
- Set appropriate `max_tokens` limits 
- Configure cron schedules to avoid unnecessary runs
- Implement error handling to prevent API waste

## Environment Variables

Add to `wrangler.toml`:

```toml
[env.production.vars]
SWARMS_API_KEY = "your-swarms-api-key"

# Stock API Keys (get free keys from these providers)
STOCK_API_KEY = "your-alpha-vantage-key"  # Free: https://www.alphavantage.co/support/#api-key
FMP_API_KEY = "your-fmp-key"              # Free: https://financialmodelingprep.com/developer/docs

# Email Configuration (Mailgun)
MAILGUN_API_KEY = "your-mailgun-api-key"  # Free: https://www.mailgun.com/
MAILGUN_DOMAIN = "your-domain.com"
RECIPIENT_EMAIL = "investor@yourcompany.com"
MEDICAL_TEAM_EMAIL = "medical-team@hospital.com"
```

## API Endpoints Used

### Stock Data APIs
- **Alpha Vantage**: Real-time stock prices, technical indicators (RSI, MACD)
- **Financial Modeling Prep**: Market news, earnings data, company fundamentals
- **Free Tier Limits**: Alpha Vantage (5 calls/min), FMP (250 calls/day)

### Real Market Data Flow
1. **Fetch Live Data**: Current prices, volume, technical indicators
2. **Get Market News**: Recent earnings, economic events, analyst reports  
3. **AI Analysis**: Swarms agents analyze real data for actionable insights
4. **Email Reports**: Professional HTML emails with analysis and data tables

### Email Features
- **Stock Reports**: Daily market analysis with data tables and key movers
- **Healthcare Alerts**: Color-coded severity levels with immediate attention flags
- **HTML Formatting**: Professional email templates with styling
- **Mailgun Integration**: Reliable email delivery service
