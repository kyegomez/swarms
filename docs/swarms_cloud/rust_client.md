# Swarms Client - Production Grade Rust SDK

A high-performance, production-ready Rust client for the Swarms API with comprehensive features for building multi-agent AI systems.

## Features

- **🚀 High Performance**: Built with `reqwest` and `tokio` for maximum throughput
- **🔄 Connection Pooling**: Automatic HTTP connection reuse and pooling
- **⚡ Circuit Breaker**: Automatic failure detection and recovery
- **💾 Intelligent Caching**: TTL-based in-memory caching with concurrent access
- **📊 Rate Limiting**: Configurable concurrent request limits
- **🔄 Retry Logic**: Exponential backoff with jitter
- **📝 Comprehensive Logging**: Structured logging with `tracing`
- **✅ Type Safety**: Full compile-time type checking with `serde`

## Installation

Install `swarms-rs` globally using cargo:

```bash
cargo install swarms-rs
```





## Quick Start

```rust
use swarms_client::SwarmsClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the client with API key from environment
    let client = SwarmsClient::builder()
        .unwrap()
        .from_env()?  // Loads API key from SWARMS_API_KEY environment variable
        .timeout(std::time::Duration::from_secs(60))
        .max_retries(3)
        .build()?;

    // Make a simple swarm completion request
    let response = client.swarm()
        .completion()
        .name("My First Swarm")
        .swarm_type(SwarmType::Auto)
        .task("Analyze the pros and cons of quantum computing")
        .agent(|agent| {
            agent
                .name("Researcher")
                .description("Conducts in-depth research")
                .model("gpt-4o")
        })
        .send()
        .await?;

    println!("Swarm output: {}", response.output);
    Ok(())
}
```

## API Reference

### SwarmsClient

The main client for interacting with the Swarms API.

#### Constructor Methods

##### `SwarmsClient::builder()`

Creates a new client builder for configuring the client.

**Returns**: `Result<ClientBuilder, SwarmsError>`

**Example**:
```rust
let client = SwarmsClient::builder()
    .unwrap()
    .api_key("your-api-key")
    .timeout(Duration::from_secs(60))
    .build()?;
```

##### `SwarmsClient::with_config(config: ClientConfig)`

Creates a client with custom configuration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ClientConfig` | Client configuration settings |

**Returns**: `Result<SwarmsClient, SwarmsError>`

**Example**:
```rust
let config = ClientConfig {
    api_key: "your-api-key".to_string(),
    base_url: "https://api.swarms.com/".parse().unwrap(),
    timeout: Duration::from_secs(120),
    max_retries: 5,
    ..Default::default()
};

let client = SwarmsClient::with_config(config)?;
```

#### Resource Access Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `agent()` | `AgentResource` | Access agent-related operations |
| `swarm()` | `SwarmResource` | Access swarm-related operations |
| `models()` | `ModelsResource` | Access model listing operations |
| `logs()` | `LogsResource` | Access logging operations |

#### Cache Management Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `clear_cache()` | None | `()` | Clears all cached responses |
| `cache_stats()` | None | `Option<(usize, usize)>` | Returns (valid_entries, total_entries) |

### ClientBuilder

Builder for configuring the Swarms client.

#### Configuration Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `new()` | None | `ClientBuilder` | Creates a new builder with defaults |
| `from_env()` | None | `Result<ClientBuilder, SwarmsError>` | Loads API key from environment |
| `api_key(key)` | `String` | `ClientBuilder` | Sets the API key |
| `base_url(url)` | `&str` | `Result<ClientBuilder, SwarmsError>` | Sets the base URL |
| `timeout(duration)` | `Duration` | `ClientBuilder` | Sets request timeout |
| `max_retries(count)` | `usize` | `ClientBuilder` | Sets maximum retry attempts |
| `retry_delay(duration)` | `Duration` | `ClientBuilder` | Sets retry delay duration |
| `max_concurrent_requests(count)` | `usize` | `ClientBuilder` | Sets concurrent request limit |
| `enable_cache(enabled)` | `bool` | `ClientBuilder` | Enables/disables caching |
| `cache_ttl(duration)` | `Duration` | `ClientBuilder` | Sets cache TTL |
| `build()` | None | `Result<SwarmsClient, SwarmsError>` | Builds the client |

**Example**:
```rust
let client = SwarmsClient::builder()
    .unwrap()
    .from_env()?
    .timeout(Duration::from_secs(120))
    .max_retries(5)
    .max_concurrent_requests(50)
    .enable_cache(true)
    .cache_ttl(Duration::from_secs(600))
    .build()?;
```

### SwarmResource

Resource for swarm-related operations.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `completion()` | None | `SwarmCompletionBuilder` | Creates a new swarm completion builder |
| `create(request)` | `SwarmSpec` | `Result<SwarmCompletionResponse, SwarmsError>` | Creates a swarm completion directly |
| `create_batch(requests)` | `Vec<SwarmSpec>` | `Result<Vec<SwarmCompletionResponse>, SwarmsError>` | Creates multiple swarm completions |
| `list_types()` | None | `Result<SwarmTypesResponse, SwarmsError>` | Lists available swarm types |

### SwarmCompletionBuilder

Builder for creating swarm completion requests.

#### Configuration Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `name(name)` | `String` | `SwarmCompletionBuilder` | Sets the swarm name |
| `description(desc)` | `String` | `SwarmCompletionBuilder` | Sets the swarm description |
| `swarm_type(type)` | `SwarmType` | `SwarmCompletionBuilder` | Sets the swarm type |
| `task(task)` | `String` | `SwarmCompletionBuilder` | Sets the main task |
| `agent(builder_fn)` | `Fn(AgentSpecBuilder) -> AgentSpecBuilder` | `SwarmCompletionBuilder` | Adds an agent using a builder function |
| `max_loops(count)` | `u32` | `SwarmCompletionBuilder` | Sets maximum execution loops |
| `service_tier(tier)` | `String` | `SwarmCompletionBuilder` | Sets the service tier |
| `send()` | None | `Result<SwarmCompletionResponse, SwarmsError>` | Sends the request |

### AgentResource

Resource for agent-related operations.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `completion()` | None | `AgentCompletionBuilder` | Creates a new agent completion builder |
| `create(request)` | `AgentCompletion` | `Result<AgentCompletionResponse, SwarmsError>` | Creates an agent completion directly |
| `create_batch(requests)` | `Vec<AgentCompletion>` | `Result<Vec<AgentCompletionResponse>, SwarmsError>` | Creates multiple agent completions |

### AgentCompletionBuilder

Builder for creating agent completion requests.

#### Configuration Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `agent_name(name)` | `String` | `AgentCompletionBuilder` | Sets the agent name |
| `task(task)` | `String` | `AgentCompletionBuilder` | Sets the task |
| `model(model)` | `String` | `AgentCompletionBuilder` | Sets the AI model |
| `description(desc)` | `String` | `AgentCompletionBuilder` | Sets the agent description |
| `system_prompt(prompt)` | `String` | `AgentCompletionBuilder` | Sets the system prompt |
| `temperature(temp)` | `f32` | `AgentCompletionBuilder` | Sets the temperature (0.0-1.0) |
| `max_tokens(tokens)` | `u32` | `AgentCompletionBuilder` | Sets maximum tokens |
| `max_loops(loops)` | `u32` | `AgentCompletionBuilder` | Sets maximum loops |
| `send()` | None | `Result<AgentCompletionResponse, SwarmsError>` | Sends the request |

### SwarmType Enum

Available swarm types for different execution patterns.

| Variant | Description |
|---------|-------------|
| `AgentRearrange` | Agents can be rearranged based on task requirements |
| `MixtureOfAgents` | Combines multiple agents with different specializations |
| `SpreadSheetSwarm` | Organized like a spreadsheet with structured data flow |
| `SequentialWorkflow` | Agents execute in a sequential order |
| `ConcurrentWorkflow` | Agents execute concurrently |
| `GroupChat` | Agents interact in a group chat format |
| `MultiAgentRouter` | Routes tasks between multiple agents |
| `AutoSwarmBuilder` | Automatically builds swarm structure |
| `HiearchicalSwarm` | Hierarchical organization of agents |
| `Auto` | Automatically selects the best swarm type |
| `MajorityVoting` | Agents vote on decisions |
| `Malt` | Multi-Agent Language Tasks |

## Detailed Examples

### 1. Simple Agent Completion

```rust
use swarms_client::{SwarmsClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = SwarmsClient::builder()
        .unwrap()
        .from_env()?
        .build()?;

    let response = client.agent()
        .completion()
        .agent_name("Content Writer")
        .task("Write a blog post about sustainable technology")
        .model("gpt-4o")
        .temperature(0.7)
        .max_tokens(2000)
        .description("An expert content writer specializing in technology topics")
        .system_prompt("You are a professional content writer with expertise in technology and sustainability. Write engaging, informative content that is well-structured and SEO-friendly.")
        .send()
        .await?;

    println!("Agent Response: {}", response.outputs);
    println!("Tokens Used: {}", response.usage.total_tokens);
    
    Ok(())
}
```

### 2. Multi-Agent Research Swarm

```rust
use swarms_client::{SwarmsClient, SwarmType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = SwarmsClient::builder()
        .unwrap()
        .from_env()?
        .timeout(Duration::from_secs(300)) // 5 minutes for complex tasks
        .build()?;

    let response = client.swarm()
        .completion()
        .name("AI Research Swarm")
        .description("A comprehensive research team analyzing AI trends and developments")
        .swarm_type(SwarmType::SequentialWorkflow)
        .task("Conduct a comprehensive analysis of the current state of AI in healthcare, including recent developments, challenges, and future prospects")
        
        // Data Collection Agent
        .agent(|agent| {
            agent
                .name("Data Collector")
                .description("Gathers comprehensive data and recent developments")
                .model("gpt-4o")
                .system_prompt("You are a research data collector specializing in AI and healthcare. Your job is to gather the most recent and relevant information about AI applications in healthcare, including clinical trials, FDA approvals, and industry developments.")
                .temperature(0.3)
                .max_tokens(3000)
        })
        
        // Technical Analyst
        .agent(|agent| {
            agent
                .name("Technical Analyst")
                .description("Analyzes technical aspects and implementation details")
                .model("gpt-4o")
                .system_prompt("You are a technical analyst with deep expertise in AI/ML technologies. Analyze the technical feasibility, implementation challenges, and technological requirements of AI solutions in healthcare.")
                .temperature(0.4)
                .max_tokens(3000)
        })
        
        // Market Analyst
        .agent(|agent| {
            agent
                .name("Market Analyst")
                .description("Analyzes market trends, adoption rates, and economic factors")
                .model("gpt-4o")
                .system_prompt("You are a market research analyst specializing in healthcare technology markets. Analyze market size, growth projections, key players, investment trends, and economic factors affecting AI adoption in healthcare.")
                .temperature(0.5)
                .max_tokens(3000)
        })
        
        // Regulatory Expert
        .agent(|agent| {
            agent
                .name("Regulatory Expert")
                .description("Analyzes regulatory landscape and compliance requirements")
                .model("gpt-4o")
                .system_prompt("You are a regulatory affairs expert with deep knowledge of healthcare regulations and AI governance. Analyze regulatory challenges, compliance requirements, ethical considerations, and policy developments affecting AI in healthcare.")
                .temperature(0.3)
                .max_tokens(3000)
        })
        
        // Report Synthesizer
        .agent(|agent| {
            agent
                .name("Report Synthesizer")
                .description("Synthesizes all analyses into a comprehensive report")
                .model("gpt-4o")
                .system_prompt("You are an expert report writer and strategic analyst. Synthesize all the previous analyses into a comprehensive, well-structured executive report with clear insights, recommendations, and future outlook.")
                .temperature(0.6)
                .max_tokens(4000)
        })
        
        .max_loops(1)
        .service_tier("premium")
        .send()
        .await?;

    println!("Research Report:");
    println!("{}", response.output);
    println!("\nSwarm executed with {} agents", response.number_of_agents);
    
    Ok(())
}
```

### 3. Financial Analysis Swarm (From Example)

```rust
use swarms_client::{SwarmsClient, SwarmType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = SwarmsClient::builder()
        .unwrap()
        .from_env()?
        .timeout(Duration::from_secs(120))
        .max_retries(3)
        .build()?;

    let response = client.swarm()
        .completion()
        .name("Financial Health Analysis Swarm")
        .description("A sequential workflow of specialized financial agents analyzing company health")
        .swarm_type(SwarmType::ConcurrentWorkflow)
        .task("Analyze the financial health of Apple Inc. (AAPL) based on their latest quarterly report")
        
        // Financial Data Collector Agent
        .agent(|agent| {
            agent
                .name("Financial Data Collector")
                .description("Specializes in gathering and organizing financial data from various sources")
                .model("gpt-4o")
                .system_prompt("You are a financial data collection specialist. Your role is to gather and organize relevant financial data, including revenue, expenses, profit margins, and key financial ratios. Present the data in a clear, structured format.")
                .temperature(0.7)
                .max_tokens(2000)
        })
        
        // Financial Ratio Analyzer Agent
        .agent(|agent| {
            agent
                .name("Ratio Analyzer")
                .description("Analyzes key financial ratios and metrics")
                .model("gpt-4o")
                .system_prompt("You are a financial ratio analysis expert. Your role is to calculate and interpret key financial ratios such as P/E ratio, debt-to-equity, current ratio, and return on equity. Provide insights on what these ratios indicate about the company's financial health.")
                .temperature(0.7)
                .max_tokens(2000)
        })
        
        // Additional agents...
        .agent(|agent| {
            agent
                .name("Investment Advisor")
                .description("Provides investment recommendations based on analysis")
                .model("gpt-4o")
                .system_prompt("You are an investment advisory specialist. Your role is to synthesize the analysis from previous agents and provide clear, actionable investment recommendations. Consider both short-term and long-term investment perspectives.")
                .temperature(0.7)
                .max_tokens(2000)
        })
        
        .max_loops(1)
        .service_tier("standard")
        .send()
        .await?;

    println!("Financial Analysis Results:");
    println!("{}", response.output);
    
    Ok(())
}
```

### 4. Batch Processing

```rust
use swarms_client::{SwarmsClient, AgentCompletion, AgentSpec};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = SwarmsClient::builder()
        .unwrap()
        .from_env()?
        .max_concurrent_requests(20) // Allow more concurrent requests for batch
        .build()?;

    // Create multiple agent completion requests
    let requests = vec![
        AgentCompletion {
            agent_config: AgentSpec {
                agent_name: "Content Creator 1".to_string(),
                model_name: "gpt-4o-mini".to_string(),
                temperature: 0.7,
                max_tokens: 1000,
                ..Default::default()
            },
            task: "Write a social media post about renewable energy".to_string(),
            history: None,
        },
        AgentCompletion {
            agent_config: AgentSpec {
                agent_name: "Content Creator 2".to_string(),
                model_name: "gpt-4o-mini".to_string(),
                temperature: 0.8,
                max_tokens: 1000,
                ..Default::default()
            },
            task: "Write a social media post about electric vehicles".to_string(),
            history: None,
        },
        // Add more requests...
    ];

    // Process all requests in batch
    let responses = client.agent()
        .create_batch(requests)
        .await?;

    for (i, response) in responses.iter().enumerate() {
        println!("Response {}: {}", i + 1, response.outputs);
        println!("Tokens used: {}\n", response.usage.total_tokens);
    }
    
    Ok(())
}
```

### 5. Custom Configuration with Error Handling

```rust
use swarms_client::{SwarmsClient, SwarmsError, ClientConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Custom configuration for production use
    let config = ClientConfig {
        api_key: std::env::var("SWARMS_API_KEY")?,
        base_url: "https://swarms-api-285321057562.us-east1.run.app/".parse()?,
        timeout: Duration::from_secs(180),
        max_retries: 5,
        retry_delay: Duration::from_secs(2),
        max_concurrent_requests: 50,
        circuit_breaker_threshold: 10,
        circuit_breaker_timeout: Duration::from_secs(120),
        enable_cache: true,
        cache_ttl: Duration::from_secs(600),
    };

    let client = SwarmsClient::with_config(config)?;

    // Example with comprehensive error handling
    match client.swarm()
        .completion()
        .name("Production Swarm")
        .swarm_type(SwarmType::Auto)
        .task("Analyze market trends for Q4 2024")
        .agent(|agent| {
            agent
                .name("Market Analyst")
                .model("gpt-4o")
                .temperature(0.5)
        })
        .send()
        .await
    {
        Ok(response) => {
            println!("Success! Job ID: {}", response.job_id);
            println!("Output: {}", response.output);
        },
        Err(SwarmsError::Authentication { message, .. }) => {
            eprintln!("Authentication error: {}", message);
        },
        Err(SwarmsError::RateLimit { message, .. }) => {
            eprintln!("Rate limit exceeded: {}", message);
            // Implement backoff strategy
        },
        Err(SwarmsError::InsufficientCredits { message, .. }) => {
            eprintln!("Insufficient credits: {}", message);
        },
        Err(SwarmsError::CircuitBreakerOpen) => {
            eprintln!("Circuit breaker is open - service temporarily unavailable");
        },
        Err(e) => {
            eprintln!("Other error: {}", e);
        }
    }
    
    Ok(())
}
```

### 6. Monitoring and Observability

```rust
use swarms_client::SwarmsClient;
use tracing::{info, warn, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for observability
    tracing_subscriber::init();

    let client = SwarmsClient::builder()
        .unwrap()
        .from_env()?
        .enable_cache(true)
        .build()?;

    // Monitor cache performance
    if let Some((valid, total)) = client.cache_stats() {
        info!("Cache stats: {}/{} entries valid", valid, total);
    }

    // Make request with monitoring
    let start = std::time::Instant::now();
    
    let response = client.swarm()
        .completion()
        .name("Monitored Swarm")
        .task("Analyze system performance metrics")
        .agent(|agent| {
            agent
                .name("Performance Analyst")
                .model("gpt-4o-mini")
        })
        .send()
        .await?;

    let duration = start.elapsed();
    info!("Request completed in {:?}", duration);
    
    if duration > Duration::from_secs(30) {
        warn!("Request took longer than expected: {:?}", duration);
    }

    // Clear cache periodically in production
    client.clear_cache();
    
    Ok(())
}
```

## Error Handling

The client provides comprehensive error handling with specific error types:

### SwarmsError Types

| Error Type | Description | Recommended Action |
|------------|-------------|-------------------|
| `Authentication` | Invalid API key or authentication failure | Check API key and permissions |
| `RateLimit` | Rate limit exceeded | Implement exponential backoff |
| `InvalidRequest` | Malformed request parameters | Validate input parameters |
| `InsufficientCredits` | Not enough credits for operation | Check account balance |
| `Api` | General API error | Check API status and retry |
| `Network` | Network connectivity issues | Check internet connection |
| `Timeout` | Request timeout | Increase timeout or retry |
| `CircuitBreakerOpen` | Circuit breaker preventing requests | Wait for recovery period |
| `Serialization` | JSON serialization/deserialization error | Check data format |

### Error Handling Best Practices

```rust
use swarms_client::{SwarmsClient, SwarmsError};

async fn handle_swarm_request(client: &SwarmsClient, task: &str) -> Result<String, SwarmsError> {
    match client.swarm()
        .completion()
        .task(task)
        .agent(|agent| agent.name("Worker").model("gpt-4o-mini"))
        .send()
        .await
    {
        Ok(response) => Ok(response.output.to_string()),
        Err(SwarmsError::RateLimit { .. }) => {
            // Implement exponential backoff
            tokio::time::sleep(Duration::from_secs(5)).await;
            Err(SwarmsError::RateLimit {
                message: "Rate limited - should retry".to_string(),
                status: Some(429),
                request_id: None,
            })
        },
        Err(e) => Err(e),
    }
}
```

## Performance Features

### Connection Pooling
The client automatically manages HTTP connection pooling for optimal performance:

```rust
// Connections are automatically pooled and reused
let client = SwarmsClient::builder()
    .unwrap()
    .from_env()?
    .max_concurrent_requests(100) // Allow up to 100 concurrent requests
    .build()?;
```

### Caching
Intelligent caching reduces redundant API calls:

```rust
let client = SwarmsClient::builder()
    .unwrap()
    .from_env()?
    .enable_cache(true)
    .cache_ttl(Duration::from_secs(300)) // 5-minute TTL
    .build()?;

// GET requests are automatically cached
let models = client.models().list().await?; // First call hits API
let models_cached = client.models().list().await?; // Second call uses cache
```

### Circuit Breaker
Automatic failure detection and recovery:

```rust
let client = SwarmsClient::builder()
    .unwrap()
    .from_env()?
    .build()?;

// Circuit breaker automatically opens after 5 failures
// and recovers after 60 seconds
```

## Configuration Reference

### ClientConfig Structure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `String` | `""` | Swarms API key |
| `base_url` | `Url` | `https://swarms-api-285321057562.us-east1.run.app/` | API base URL |
| `timeout` | `Duration` | `60s` | Request timeout |
| `max_retries` | `usize` | `3` | Maximum retry attempts |
| `retry_delay` | `Duration` | `1s` | Base retry delay |
| `max_concurrent_requests` | `usize` | `100` | Concurrent request limit |
| `circuit_breaker_threshold` | `usize` | `5` | Failure threshold for circuit breaker |
| `circuit_breaker_timeout` | `Duration` | `60s` | Circuit breaker recovery time |
| `enable_cache` | `bool` | `true` | Enable response caching |
| `cache_ttl` | `Duration` | `300s` | Cache time-to-live |

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SWARMS_API_KEY` | Your Swarms API key | `sk-xxx...` |
| `SWARMS_BASE_URL` | Custom API base URL (optional) | `https://api.custom.com/` |

## Testing

Run the test suite:

```bash
cargo test
```

Run specific tests:

```bash
cargo test test_cache
cargo test test_circuit_breaker
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.