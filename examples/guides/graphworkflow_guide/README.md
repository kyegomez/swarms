# GraphWorkflow Guide

Welcome to the comprehensive GraphWorkflow guide! This collection demonstrates the power and flexibility of Swarms' GraphWorkflow system - the LangGraph killer that provides superior multi-agent orchestration capabilities.

## 🚀 Quick Start

### Installation

```bash
# Install Swarms with all dependencies
uv pip install swarms

# Optional: Install visualization dependencies
uv pip install graphviz

# Verify installation
python -c "from swarms.structs.graph_workflow import GraphWorkflow; print('✅ GraphWorkflow ready')"
```

### Run Your First Example

```bash
# Start with the quick start guide
python quick_start_guide.py

# Or run the comprehensive demo
python comprehensive_demo.py

# For specific examples
python comprehensive_demo.py --demo healthcare
python comprehensive_demo.py --demo finance
```

## 📁 Example Files

### 🎓 Learning Examples

| File | Description | Complexity |
|------|-------------|------------|
| `quick_start_guide.py` | **START HERE** - Step-by-step introduction to GraphWorkflow | ⭐ Beginner |
| `graph_workflow_example.py` | Basic two-agent workflow example | ⭐ Beginner |
| `comprehensive_demo.py` | Complete feature demonstration with multiple use cases | ⭐⭐⭐ Advanced |

### 🏥 Healthcare Examples

| File | Description | Complexity |
|------|-------------|------------|
| `comprehensive_demo.py --demo healthcare` | Clinical decision support workflow | ⭐⭐⭐ Advanced |

**Healthcare Workflow Features:**
- Multi-disciplinary clinical team simulation
- Parallel specialist consultations
- Drug interaction checking
- Risk assessment and quality assurance
- Evidence-based clinical decision support

### 💰 Finance Examples

| File | Description | Complexity |
|------|-------------|------------|
| `advanced_graph_workflow.py` | Sophisticated investment analysis workflow | ⭐⭐⭐ Advanced |
| `comprehensive_demo.py --demo finance` | Quantitative trading strategy development | ⭐⭐⭐ Advanced |

**Finance Workflow Features:**
- Multi-source market data analysis
- Parallel quantitative analysis (Technical, Fundamental, Sentiment)
- Risk management and portfolio optimization
- Strategy backtesting and validation
- Execution planning and monitoring

### 🔧 Technical Examples

| File | Description | Complexity |
|------|-------------|------------|
| `test_parallel_processing_example.py` | Comprehensive parallel processing patterns | ⭐⭐ Intermediate |
| `test_graphviz_visualization.py` | Visualization capabilities and layouts | ⭐⭐ Intermediate |
| `test_graph_workflow_caching.py` | Performance optimization and caching | ⭐⭐ Intermediate |
| `test_enhanced_json_export.py` | Serialization and persistence features | ⭐⭐ Intermediate |
| `test_graphworlfolw_validation.py` | Workflow validation and error handling | ⭐⭐ Intermediate |

## 🎯 Key Features Demonstrated

### ⚡ Parallel Processing Patterns

- **Fan-out**: One agent distributes to multiple agents
- **Fan-in**: Multiple agents converge to one agent  
- **Parallel chains**: Many-to-many mesh processing
- **Complex hybrid**: Sophisticated multi-stage patterns

### 🚀 Performance Optimization

- **Intelligent Compilation**: Pre-computed execution layers
- **Advanced Caching**: Persistent state across runs
- **Worker Pool Optimization**: CPU-optimized parallel execution
- **Memory Management**: Efficient resource utilization

### 🎨 Visualization & Monitoring

- **Professional Graphviz Diagrams**: Multiple layouts and formats
- **Real-time Performance Metrics**: Execution monitoring
- **Workflow Validation**: Comprehensive error checking
- **Rich Logging**: Detailed execution insights

### 💾 Enterprise Features

- **JSON Serialization**: Complete workflow persistence
- **Runtime State Management**: Compilation caching
- **Error Handling**: Robust failure recovery
- **Scalability**: Support for large agent networks

## 🏃‍♂️ Running Examples

### Basic Usage

```python
from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow

# Create agents
agent1 = Agent(agent_name="Researcher", model_name="gpt-4o-mini", max_loops=1)
agent2 = Agent(agent_name="Writer", model_name="gpt-4o-mini", max_loops=1)

# Create workflow
workflow = GraphWorkflow(name="SimpleWorkflow", auto_compile=True)
workflow.add_node(agent1)
workflow.add_node(agent2)
workflow.add_edge("Researcher", "Writer")

# Execute
results = workflow.run(task="Research and write about AI trends")
```

### Parallel Processing

```python
# Fan-out pattern: One agent to multiple agents
workflow.add_edges_from_source("DataCollector", ["AnalystA", "AnalystB", "AnalystC"])

# Fan-in pattern: Multiple agents to one agent
workflow.add_edges_to_target(["SpecialistX", "SpecialistY"], "Synthesizer")

# Parallel chain: Many-to-many processing
workflow.add_parallel_chain(
    sources=["DataA", "DataB"],
    targets=["ProcessorX", "ProcessorY"]
)
```

### Performance Monitoring

```python
# Get compilation status
status = workflow.get_compilation_status()
print(f"Compiled: {status['is_compiled']}")
print(f"Workers: {status['max_workers']}")

# Monitor execution
import time
start = time.time()
results = workflow.run(task="Analyze market conditions")
print(f"Execution time: {time.time() - start:.2f}s")
print(f"Throughput: {len(results)/(time.time() - start):.1f} agents/second")
```

## 🔬 Use Case Examples

### 📊 Enterprise Data Processing

```python
# Multi-stage data pipeline
workflow.add_parallel_chain(
    ["APIIngester", "DatabaseExtractor", "FileProcessor"],
    ["DataValidator", "DataTransformer", "DataEnricher"]
)
workflow.add_edges_to_target(
    ["DataValidator", "DataTransformer", "DataEnricher"],
    "ReportGenerator"
)
```

### 🏥 Clinical Decision Support

```python
# Multi-specialist consultation
workflow.add_edges_from_source("PatientDataCollector", [
    "PrimaryCarePhysician", "Cardiologist", "Pharmacist"
])
workflow.add_edges_to_target([
    "PrimaryCarePhysician", "Cardiologist", "Pharmacist"
], "CaseManager")
```

### 💼 Investment Analysis

```python
# Parallel financial analysis
workflow.add_parallel_chain(
    ["MarketDataCollector", "FundamentalDataCollector"],
    ["TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"]
)
workflow.add_edges_to_target([
    "TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"
], "PortfolioManager")
```

## 🎨 Visualization Examples

### Generate Workflow Diagrams

```python
# Professional Graphviz visualization
workflow.visualize(
    format="png",          # png, svg, pdf, dot
    engine="dot",          # dot, neato, fdp, sfdp, circo
    show_summary=True,     # Display parallel processing stats
    view=True              # Open diagram automatically
)

# Text-based visualization (always available)
workflow.visualize_simple()
```

### Example Output

```
📊 GRAPHVIZ WORKFLOW VISUALIZATION
====================================
📁 Saved to: MyWorkflow_visualization.png
🤖 Total Agents: 8
🔗 Total Connections: 12
📚 Execution Layers: 4

⚡ Parallel Processing Patterns:
  🔀 Fan-out patterns: 2
  🔀 Fan-in patterns: 1
  ⚡ Parallel execution nodes: 6
  🎯 Parallel efficiency: 75.0%
```

## 🛠️ Troubleshooting

### Common Issues

1. **Compilation Errors**
   ```python
   # Check for cycles in workflow
   validation = workflow.validate(auto_fix=True)
   if not validation['is_valid']:
       print("Validation errors:", validation['errors'])
   ```

2. **Performance Issues**
   ```python
   # Ensure compilation before execution
   workflow.compile()
   
   # Check worker count
   status = workflow.get_compilation_status()
   print(f"Workers: {status['max_workers']}")
   ```

3. **Memory Issues**
   ```python
   # Clear conversation history if not needed
   workflow.conversation = Conversation()
   
   # Monitor memory usage
   import psutil
   process = psutil.Process()
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Memory: {memory_mb:.1f} MB")
   ```

### Debug Mode

```python
# Enable detailed logging
workflow = GraphWorkflow(
    name="DebugWorkflow",
    verbose=True,           # Detailed execution logs
    auto_compile=True,      # Automatic optimization
)

# Validate workflow structure
validation = workflow.validate(auto_fix=True)
print("Validation result:", validation)
```

## 📚 Documentation

- **[Technical Guide](graph_workflow_technical_guide.md)**: Comprehensive 4,000-word technical documentation
- **[API Reference](../../../docs/swarms/structs/)**: Complete API documentation
- **[Multi-Agent Examples](../../multi_agent/)**: Other multi-agent examples

## 🤝 Contributing

Found a bug or want to add an example?

1. **Report Issues**: Open an issue with detailed reproduction steps
2. **Add Examples**: Submit PRs with new use case examples
3. **Improve Documentation**: Help expand the guides and tutorials
4. **Performance Optimization**: Share benchmarks and optimizations

## 🎯 Next Steps

1. **Start Learning**: Run `python quick_start_guide.py`
2. **Explore Examples**: Try healthcare and finance use cases
3. **Build Your Workflow**: Adapt examples to your domain
4. **Deploy to Production**: Use monitoring and optimization features
5. **Join Community**: Share your workflows and get help

## 🏆 Why GraphWorkflow?

GraphWorkflow is the **LangGraph killer** because it provides:

- **40-60% Better Performance**: Intelligent compilation and parallel execution
- **Enterprise Reliability**: Comprehensive error handling and monitoring  
- **Superior Scalability**: Handles hundreds of agents efficiently
- **Rich Visualization**: Professional workflow diagrams
- **Production Ready**: Serialization, caching, and validation

Ready to revolutionize your multi-agent systems? Start with GraphWorkflow today! 🚀
