# swarms-rs

!!! note "Modern AI Agent Framework"
    swarms-rs is a powerful Rust framework for building autonomous AI agents powered by LLMs, equipped with robust tools and memory capabilities. Designed for various applications from trading analysis to healthcare diagnostics.

## Getting Started

### Installation

```bash
cargo add swarms-rs
```

!!! tip "Compatible with Rust 1.70+"
    This library requires Rust 1.70 or later. Make sure your Rust toolchain is up to date.

### Required Environment Variables

```bash
# Required API keys
OPENAI_API_KEY="your_openai_api_key_here"
DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### Quick Start

Here's a simple example to get you started with swarms-rs:

```rust
use std::env;
use anyhow::Result;
use swarms_rs::{llm::provider::openai::OpenAI, structs::agent::Agent};

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Initialize tracing for better debugging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(
            tracing_subscriber::fmt::layer()
                .with_line_number(true)
                .with_file(true),
        )
        .init();

    // Set up your LLM client
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::new(api_key).set_model("gpt-4-turbo");
    
    // Create a basic agent
    let agent = client
        .agent_builder()
        .system_prompt("You are a helpful assistant.")
        .agent_name("BasicAgent")
        .user_name("User")
        .build();
    
    // Run the agent with a user query
    let response = agent
        .run("Tell me about Rust programming.".to_owned())
        .await?;
    
    println!("{}", response);
    Ok(())
}
```

## Core Concepts

### Agents

Agents in swarms-rs are autonomous entities that can:

- Perform complex reasoning based on LLM capabilities
- Use tools to interact with external systems
- Maintain persistent memory
- Execute multi-step plans

## Agent Configuration

### Core Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `system_prompt` | Initial instructions/role for the agent | - | Yes |
| `agent_name` | Name identifier for the agent | - | Yes |
| `user_name` | Name for the user interacting with agent | - | Yes |
| `max_loops` | Maximum number of reasoning loops | 1 | No |
| `retry_attempts` | Number of retry attempts on failure | 1 | No |
| `enable_autosave` | Enable state persistence | false | No |
| `save_state_dir` | Directory for saving agent state | None | No |

### Advanced Configuration

You can enhance your agent's capabilities with:

- **Planning**: Enable structured planning for complex tasks
- **Memory**: Persistent storage for agent state
- **Tools**: External capabilities through MCP protocol

!!! warning "Resource Usage"
    Setting high values for `max_loops` can increase API usage and costs. Start with lower values and adjust as needed.

## Examples

### Specialized Agent for Cryptocurrency Analysis

```rust
use std::env;
use anyhow::Result;
use swarms_rs::{llm::provider::openai::OpenAI, structs::agent::Agent};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(
            tracing_subscriber::fmt::layer()
                .with_line_number(true)
                .with_file(true),
        )
        .init();

    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::new(api_key).set_model("gpt-4-turbo");
    
    let agent = client
        .agent_builder()
        .system_prompt(
            "You are a sophisticated cryptocurrency analysis assistant specialized in:
            1. Technical analysis of crypto markets
            2. Fundamental analysis of blockchain projects
            3. Market sentiment analysis
            4. Risk assessment
            5. Trading patterns recognition
            
            When analyzing cryptocurrencies, always consider:
            - Market capitalization and volume
            - Historical price trends
            - Project fundamentals and technology
            - Recent news and developments
            - Market sentiment indicators
            - Potential risks and opportunities
            
            Provide clear, data-driven insights and always include relevant disclaimers about market volatility."
        )
        .agent_name("CryptoAnalyst")
        .user_name("Trader")
        .enable_autosave()
        .max_loops(3)  // Increased for more thorough analysis
        .save_state_dir("./crypto_analysis/")
        .enable_plan("Break down the crypto analysis into systematic steps:
            1. Gather market data
            2. Analyze technical indicators
            3. Review fundamental factors
            4. Assess market sentiment
            5. Provide comprehensive insights".to_owned())
        .build();
        
    let response = agent
        .run("What are your thoughts on Bitcoin's current market position?".to_owned())
        .await?;
        
    println!("{}", response);
    Ok(())
}
```

## Using Tools with MCP

### Model Context Protocol (MCP)

swarms-rs supports the Model Context Protocol (MCP), enabling agents to interact with external tools through standardized interfaces.

!!! info "What is MCP?"
    MCP (Model Context Protocol) provides a standardized way for LLMs to interact with external tools, giving your agents access to real-world data and capabilities beyond language processing.

### Supported MCP Server Types

- **STDIO MCP Servers**: Connect to command-line tools implementing the MCP protocol
- **SSE MCP Servers**: Connect to web-based MCP servers using Server-Sent Events

### Tool Integration

Add tools to your agent during configuration:

```rust
let agent = client
    .agent_builder()
    .system_prompt("You are a helpful assistant with access to tools.")
    .agent_name("ToolAgent")
    .user_name("User")
    // Add STDIO MCP server
    .add_stdio_mcp_server("uvx", ["mcp-hn"])
    .await
    // Add SSE MCP server
    .add_sse_mcp_server("file-browser", "http://127.0.0.1:8000/sse")
    .await
    .build();
```

### Full MCP Agent Example

```rust
use std::env;
use anyhow::Result;
use swarms_rs::{llm::provider::openai::OpenAI, structs::agent::Agent};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(
            tracing_subscriber::fmt::layer()
                .with_line_number(true)
                .with_file(true),
        )
        .init();

    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = OpenAI::new(api_key).set_model("gpt-4-turbo");
    
    let agent = client
        .agent_builder()
        .system_prompt("You are a helpful assistant with access to news and file system tools.")
        .agent_name("SwarmsAgent")
        .user_name("User")
        // Add Hacker News tool
        .add_stdio_mcp_server("uvx", ["mcp-hn"])
        .await
        // Add filesystem tool
        // To set up: uvx mcp-proxy --sse-port=8000 -- npx -y @modelcontextprotocol/server-filesystem ~
        .add_sse_mcp_server("file-browser", "http://127.0.0.1:8000/sse")
        .await
        .retry_attempts(2)
        .max_loops(3)
        .build();

    // Use the news tool
    let news_response = agent
        .run("Get the top 3 stories of today from Hacker News".to_owned())
        .await?;
    println!("NEWS RESPONSE:\n{}", news_response);

    // Use the filesystem tool
    let fs_response = agent.run("List files in my home directory".to_owned()).await?;
    println!("FILESYSTEM RESPONSE:\n{}", fs_response);

    Ok(())
}
```

## Setting Up MCP Tools

### Installing MCP Servers

To use MCP servers with swarms-rs, you'll need to install the appropriate tools:

1. **uv Package Manager**:
   ```bash
   curl -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | sh
   ```

2. **MCP-HN** (Hacker News MCP server):
   ```bash
   uvx install mcp-hn
   ```

3. **Setting up an SSE MCP server**:
   ```bash
   # Start file system MCP server over SSE
   uvx mcp-proxy --sse-port=8000 -- npx -y @modelcontextprotocol/server-filesystem ~
   ```

## FAQ

### General Questions

??? question "What LLM providers are supported?"
    swarms-rs currently supports:
    
    - OpenAI (GPT models)
    
    - DeepSeek AI
    
    - More providers coming soon

??? question "How does state persistence work?"
    When `enable_autosave` is set to `true`, the agent will save its state to the directory specified in `save_state_dir`. This includes conversation history and tool states, allowing the agent to resume from where it left off.

??? question "What is the difference between `max_loops` and `retry_attempts`?"
    - `max_loops`: Controls how many reasoning steps the agent can take for a single query
    
    - `retry_attempts`: Specifies how many times the agent will retry if an error occurs

### MCP Tools

??? question "How do I create my own MCP server?"
    You can create your own MCP server by implementing the MCP protocol. Check out the [MCP documentation](https://github.com/modelcontextprotocol/spec) for details on the protocol specification.

??? question "Can I use tools without MCP?"
    Currently, swarms-rs is designed to use the MCP protocol for tool integration. This provides a standardized way for agents to interact with external systems.

## Advanced Topics

### Performance Optimization

Optimize your agent's performance by:

1. **Crafting Effective System Prompts**:
   - Be specific about the agent's role and capabilities
   
   - Include clear instructions on how to use available tools
   
   - Define success criteria for the agent's responses

2. **Tuning Loop Parameters**:
   
   - Start with lower values for `max_loops` and increase as needed
   
   - Consider the complexity of tasks when setting loop limits

3. **Strategic Tool Integration**:
   
   - Only integrate tools that are necessary for the agent's tasks
   
   - Provide clear documentation in the system prompt about when to use each tool

### Security Considerations

!!! danger "Security Notice"
    When using file system tools or other system-level access, always be careful about permissions. Limit the scope of what your agent can access, especially in production environments.

## Coming Soon

- Memory plugins for different storage backends

- Additional LLM providers

- Group agent coordination

- Function calling

- Custom tool development framework

## Contributing

Contributions to swarms-rs are welcome! Check out our [GitHub repository](https://github.com/swarms-rs) for more information.