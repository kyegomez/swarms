# Cloudflare Workers Deployment with Swarms API

Deploy intelligent agent swarms on Cloudflare's edge network for maximum performance, global availability, and automatic scaling. This guide covers deploying both API endpoints and scheduled cron job agents using the Swarms API.

## Overview

Cloudflare Workers provide a serverless execution environment that runs on Cloudflare's global network, offering:

- **Edge Computing**: Deploy agents close to users worldwide
- **Automatic Scaling**: Handle traffic spikes without configuration
- **Zero Cold Starts**: Instant response times
- **Cost Effective**: Pay only for what you use
- **Built-in Cron Jobs**: Schedule automated agent tasks

Perfect for deploying AI agents that need global reach, low latency, and scheduled execution capabilities.

## Prerequisites

!!! info "Requirements"

    - Cloudflare account (free tier available)
    - Wrangler CLI installed (`npm install -g wrangler`)
    - Swarms API key from [Swarms Platform](https://swarms.world/platform/api-keys)
    - Node.js 16+ for local development

## Quick Start

### 1. Project Setup

```bash
# Create new Cloudflare Worker
npx create-cloudflare my-swarms-worker worker

# Navigate to project
cd my-swarms-worker

# Install dependencies
npm install
```

### 2. Configure Environment Variables

Add your Swarms API key to `wrangler.toml`:

```toml
name = "my-swarms-worker"
main = "src/index.js"
compatibility_date = "2024-01-01"

[env.production.vars]
SWARMS_API_KEY = "your-api-key-here"

[env.production]
# For scheduled agents
[[env.production.triggers.crons]]
cron = "0 9 * * MON-FRI"  # Weekdays at 9 AM UTC
```

### 3. Basic Agent Worker

=== "JavaScript"

    ```javascript
    export default {
      async fetch(request, env, ctx) {
        // Handle CORS preflight
        if (request.method === 'OPTIONS') {
          return handleCORS();
        }

        try {
          const { pathname } = new URL(request.url);
          
          if (pathname === '/api/agent' && request.method === 'POST') {
            return await handleAgentRequest(request, env);
          }
          
          if (pathname === '/api/health' && request.method === 'GET') {
            return new Response(JSON.stringify({ 
              status: 'healthy', 
              service: 'Swarms Agent API',
              version: '1.0.0'
            }), {
              status: 200,
              headers: { 
                'Content-Type': 'application/json',
                ...getCORSHeaders()
              }
            });
          }
          
          return new Response('Not Found', { status: 404 });
        } catch (error) {
          console.error('Worker error:', error);
          return new Response(JSON.stringify({ 
            error: error.message 
          }), {
            status: 500,
            headers: { 
              'Content-Type': 'application/json',
              ...getCORSHeaders()
            }
          });
        }
      },

      // Scheduled event handler for cron jobs
      async scheduled(event, env, ctx) {
        ctx.waitUntil(handleScheduledEvent(event, env));
      }
    };

    async function handleAgentRequest(request, env) {
      const requestData = await request.json();
      
      const agentConfig = {
        agent_config: {
          agent_name: requestData.agent_name || "CloudflareAgent",
          description: requestData.description || "Agent running on Cloudflare Workers",
          system_prompt: requestData.system_prompt || "You are a helpful AI assistant.",
          model_name: requestData.model_name || "gpt-4o-mini",
          max_tokens: requestData.max_tokens || 2000,
          temperature: requestData.temperature || 0.7
        },
        task: requestData.task
      };

      const response = await fetch('https://api.swarms.world/v1/agent/completions', {
        method: 'POST',
        headers: {
          'x-api-key': env.SWARMS_API_KEY,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(agentConfig)
      });

      const result = await response.json();
      
      return new Response(JSON.stringify(result), {
        status: response.status,
        headers: {
          'Content-Type': 'application/json',
          ...getCORSHeaders()
        }
      });
    }

    function getCORSHeaders() {
      return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      };
    }

    function handleCORS() {
      return new Response(null, {
        status: 200,
        headers: getCORSHeaders()
      });
    }
    ```

=== "TypeScript"

    ```typescript
    interface AgentRequest {
      agent_name?: string;
      description?: string;
      system_prompt?: string;
      model_name?: string;
      max_tokens?: number;
      temperature?: number;
      task: string;
    }

    interface AgentConfig {
      agent_config: {
        agent_name: string;
        description: string;
        system_prompt: string;
        model_name: string;
        max_tokens: number;
        temperature: number;
      };
      task: string;
    }

    interface Env {
      SWARMS_API_KEY: string;
    }

    export interface ScheduledEvent {
      scheduledTime: number;
      cron: string;
    }

    export default {
      async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
        if (request.method === 'OPTIONS') {
          return handleCORS();
        }

        try {
          const { pathname } = new URL(request.url);
          
          if (pathname === '/api/agent' && request.method === 'POST') {
            return await handleAgentRequest(request, env);
          }
          
          if (pathname === '/api/health' && request.method === 'GET') {
            return new Response(JSON.stringify({ 
              status: 'healthy', 
              service: 'Swarms Agent API',
              version: '1.0.0'
            }), {
              status: 200,
              headers: { 
                'Content-Type': 'application/json',
                ...getCORSHeaders()
              }
            });
          }
          
          return new Response('Not Found', { status: 404 });
        } catch (error) {
          console.error('Worker error:', error);
          return new Response(JSON.stringify({ 
            error: (error as Error).message 
          }), {
            status: 500,
            headers: { 
              'Content-Type': 'application/json',
              ...getCORSHeaders()
            }
          });
        }
      },

      async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
        ctx.waitUntil(handleScheduledEvent(event, env));
      }
    };

    async function handleAgentRequest(request: Request, env: Env): Promise<Response> {
      const requestData: AgentRequest = await request.json();
      
      const agentConfig: AgentConfig = {
        agent_config: {
          agent_name: requestData.agent_name || "CloudflareAgent",
          description: requestData.description || "Agent running on Cloudflare Workers",
          system_prompt: requestData.system_prompt || "You are a helpful AI assistant.",
          model_name: requestData.model_name || "gpt-4o-mini",
          max_tokens: requestData.max_tokens || 2000,
          temperature: requestData.temperature || 0.7
        },
        task: requestData.task
      };

      const response = await fetch('https://api.swarms.world/v1/agent/completions', {
        method: 'POST',
        headers: {
          'x-api-key': env.SWARMS_API_KEY,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(agentConfig)
      });

      const result = await response.json();
      
      return new Response(JSON.stringify(result), {
        status: response.status,
        headers: {
          'Content-Type': 'application/json',
          ...getCORSHeaders()
        }
      });
    }

    function getCORSHeaders(): Record<string, string> {
      return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      };
    }

    function handleCORS(): Response {
      return new Response(null, {
        status: 200,
        headers: getCORSHeaders()
      });
    }
    ```

### 4. Deploy Your Worker

```bash
# Deploy to Cloudflare
wrangler deploy

# View logs
wrangler tail
```

## Scheduled Cron Job Agents

Cloudflare Workers support scheduled events (cron jobs) that can trigger agent tasks automatically. This is perfect for periodic analysis, monitoring, reporting, and automated decision-making.

### Stock Market Analysis Agent

Deploy a cron job agent that analyzes stock market trends and generates daily reports:

=== "JavaScript"

    ```javascript
    // wrangler.toml configuration for stock market agent
    /*
    [[env.production.triggers.crons]]
    cron = "0 16 * * MON-FRI"  # 4 PM UTC (after US market close)
    */

    async function handleScheduledEvent(event, env) {
      console.log('Stock market analysis triggered at:', new Date(event.scheduledTime));
      
      try {
        // Stock analysis swarm configuration
        const swarmConfig = {
          name: "Stock Market Analysis Swarm",
          description: "Daily stock market analysis and trading insights",
          agents: [
            {
              agent_name: "Market Data Analyst",
              description: "Analyzes market trends and price movements",
              system_prompt: `You are an expert financial market analyst specializing in equity markets. 
                Analyze market trends, volume patterns, and price movements. 
                Provide technical analysis insights and identify key support/resistance levels.
                Focus on major indices (S&P 500, NASDAQ, DOW) and sector performance.`,
              model_name: "gpt-4o",
              max_tokens: 3000,
              temperature: 0.3,
              role: "worker"
            },
            {
              agent_name: "Economic News Analyzer",
              description: "Analyzes economic news impact on markets",
              system_prompt: `You are a financial news analyst with expertise in market sentiment analysis.
                Analyze recent economic news, earnings reports, and market-moving events.
                Assess their potential impact on stock prices and market sentiment.
                Identify key catalysts and risk factors for the next trading day.`,
              model_name: "gpt-4o",
              max_tokens: 3000,
              temperature: 0.3,
              role: "worker"  
            },
            {
              agent_name: "Trading Strategy Advisor",
              description: "Provides trading recommendations and risk assessment",
              system_prompt: `You are a quantitative trading strategist with expertise in risk management.
                Based on technical analysis and market sentiment, provide actionable trading insights.
                Include risk assessment, position sizing recommendations, and key levels to watch.
                Focus on risk-adjusted returns and downside protection strategies.`,
              model_name: "gpt-4o",
              max_tokens: 3000,
              temperature: 0.4,
              role: "worker"
            }
          ],
          swarm_type: "SequentialWorkflow",
          max_loops: 1,
          task: `Analyze today's stock market performance and provide insights for tomorrow's trading session. 
            Include: 1) Market overview and key movers 2) Technical analysis of major indices 
            3) Economic news impact assessment 4) Trading opportunities and risks 
            5) Key levels to watch tomorrow. Format as a professional daily market report.`,
          service_tier: "standard"
        };

        // Execute the swarm
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
          console.log('Stock analysis completed successfully');
          console.log('Cost:', result.metadata?.billing_info?.total_cost);
          
          // Store results in KV storage for retrieval
          if (env.STOCK_REPORTS) {
            const reportKey = `stock-report-${new Date().toISOString().split('T')[0]}`;
            await env.STOCK_REPORTS.put(reportKey, JSON.stringify({
              timestamp: new Date().toISOString(),
              analysis: result.output,
              cost: result.metadata?.billing_info?.total_cost
            }));
          }
          
          // Optional: Send to webhook, email, or notification service
          await sendNotification(env, result.output);
          
        } else {
          console.error('Stock analysis failed:', result);
        }
        
      } catch (error) {
        console.error('Error in scheduled stock analysis:', error);
      }
    }

    async function sendNotification(env, analysis) {
      // Example: Send to Slack webhook
      if (env.SLACK_WEBHOOK_URL) {
        try {
          await fetch(env.SLACK_WEBHOOK_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: "📈 Daily Stock Market Analysis Ready",
              attachments: [{
                color: "good",
                text: typeof analysis === 'string' ? analysis.substring(0, 500) + '...' : 'Analysis completed',
                footer: "Swarms Stock Analysis Agent"
              }]
            })
          });
        } catch (error) {
          console.error('Failed to send Slack notification:', error);
        }
      }
    }

    // API endpoint to retrieve stock reports
    async function getStockReport(request, env) {
      const url = new URL(request.url);
      const date = url.searchParams.get('date') || new Date().toISOString().split('T')[0];
      const reportKey = `stock-report-${date}`;
      
      if (env.STOCK_REPORTS) {
        const report = await env.STOCK_REPORTS.get(reportKey);
        if (report) {
          return new Response(report, {
            headers: { 'Content-Type': 'application/json' }
          });
        }
      }
      
      return new Response(JSON.stringify({ error: 'Report not found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    ```

=== "TypeScript"

    ```typescript
    interface StockAnalysisResult {
      timestamp: string;
      analysis: any;
      cost: number;
    }

    async function handleScheduledEvent(event: ScheduledEvent, env: Env): Promise<void> {
      console.log('Stock market analysis triggered at:', new Date(event.scheduledTime));
      
      try {
        const swarmConfig = {
          name: "Stock Market Analysis Swarm",
          description: "Daily stock market analysis and trading insights",
          agents: [
            {
              agent_name: "Market Data Analyst",
              description: "Analyzes market trends and price movements",
              system_prompt: `You are an expert financial market analyst specializing in equity markets. 
                Analyze market trends, volume patterns, and price movements. 
                Provide technical analysis insights and identify key support/resistance levels.
                Focus on major indices (S&P 500, NASDAQ, DOW) and sector performance.`,
              model_name: "gpt-4o",
              max_tokens: 3000,
              temperature: 0.3,
              role: "worker"
            },
            {
              agent_name: "Economic News Analyzer", 
              description: "Analyzes economic news impact on markets",
              system_prompt: `You are a financial news analyst with expertise in market sentiment analysis.
                Analyze recent economic news, earnings reports, and market-moving events.
                Assess their potential impact on stock prices and market sentiment.
                Identify key catalysts and risk factors for the next trading day.`,
              model_name: "gpt-4o",
              max_tokens: 3000,
              temperature: 0.3,
              role: "worker"
            },
            {
              agent_name: "Trading Strategy Advisor",
              description: "Provides trading recommendations and risk assessment", 
              system_prompt: `You are a quantitative trading strategist with expertise in risk management.
                Based on technical analysis and market sentiment, provide actionable trading insights.
                Include risk assessment, position sizing recommendations, and key levels to watch.
                Focus on risk-adjusted returns and downside protection strategies.`,
              model_name: "gpt-4o",
              max_tokens: 3000,
              temperature: 0.4,
              role: "worker"
            }
          ],
          swarm_type: "SequentialWorkflow" as const,
          max_loops: 1,
          task: `Analyze today's stock market performance and provide insights for tomorrow's trading session. 
            Include: 1) Market overview and key movers 2) Technical analysis of major indices 
            3) Economic news impact assessment 4) Trading opportunities and risks 
            5) Key levels to watch tomorrow. Format as a professional daily market report.`,
          service_tier: "standard"
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
          console.log('Stock analysis completed successfully');
          console.log('Cost:', result.metadata?.billing_info?.total_cost);
          
          if (env.STOCK_REPORTS) {
            const reportKey = `stock-report-${new Date().toISOString().split('T')[0]}`;
            const reportData: StockAnalysisResult = {
              timestamp: new Date().toISOString(),
              analysis: result.output,
              cost: result.metadata?.billing_info?.total_cost
            };
            
            await env.STOCK_REPORTS.put(reportKey, JSON.stringify(reportData));
          }
          
          await sendNotification(env, result.output);
          
        } else {
          console.error('Stock analysis failed:', result);
        }
        
      } catch (error) {
        console.error('Error in scheduled stock analysis:', error);
      }
    }

    async function sendNotification(env: Env, analysis: any): Promise<void> {
      if (env.SLACK_WEBHOOK_URL) {
        try {
          await fetch(env.SLACK_WEBHOOK_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: "📈 Daily Stock Market Analysis Ready",
              attachments: [{
                color: "good",
                text: typeof analysis === 'string' ? analysis.substring(0, 500) + '...' : 'Analysis completed',
                footer: "Swarms Stock Analysis Agent"
              }]
            })
          });
        } catch (error) {
          console.error('Failed to send Slack notification:', error);
        }
      }
    }
    ```

