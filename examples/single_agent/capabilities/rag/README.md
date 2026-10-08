# Qdrant RAG Example with Document Ingestion

This example demonstrates how to use the agent structure from `example.py` with Qdrant RAG to ingest a vast array of PDF documents and text files for advanced quantitative trading analysis.

## 🚀 Features

- **Document Ingestion**: Process PDF, TXT, and Markdown files automatically
- **Qdrant Vector Database**: High-performance vector storage with similarity search
- **Sentence Transformer Embeddings**: Local embedding generation using state-of-the-art models
- **Intelligent Chunking**: Smart text chunking with overlap for better retrieval
- **Concurrent Processing**: Multi-threaded document processing for large collections
- **RAG Integration**: Seamless integration with Swarms Agent framework
- **Financial Analysis**: Specialized for quantitative trading and financial research

## 📋 Prerequisites

- Python 3.10+
- Qdrant client (local or cloud)
- Sentence transformers for embeddings
- Swarms framework

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (optional, for cloud deployment):
   ```bash
   export QDRANT_URL="your_qdrant_url"
   export QDRANT_API_KEY="your_qdrant_api_key"
   ```

## 🏗️ Architecture

The example consists of three main components:

### 1. DocumentProcessor
- Handles file discovery and text extraction
- Supports PDF, TXT, and Markdown formats
- Concurrent processing for large document collections
- Error handling and validation

### 2. QdrantRAGMemory
- Vector database management with Qdrant
- Intelligent text chunking with overlap
- Semantic search capabilities
- Metadata storage and retrieval

### 3. QuantitativeTradingRAGAgent
- Combines Swarms Agent with RAG capabilities
- Financial analysis specialization
- Document context enhancement
- Query processing and response generation

## 📖 Usage

### Basic Setup

```python
from qdrant_rag_example import QuantitativeTradingRAGAgent

# Initialize the agent
agent = QuantitativeTradingRAGAgent(
    agent_name="Financial-Analysis-Agent",
    collection_name="financial_documents",
    model_name="claude-sonnet-4-20250514"
)
```

### Document Ingestion

```python
# Ingest documents from a directory
documents_path = "./financial_documents"
num_ingested = agent.ingest_documents(documents_path)
print(f"Ingested {num_ingested} documents")
```

### Querying Documents

```python
# Search for relevant information
results = agent.query_documents("gold ETFs investment strategies", limit=5)
for result in results:
    print(f"Document: {result['document_name']}")
    print(f"Relevance: {result['similarity_score']:.3f}")
    print(f"Content: {result['chunk_text'][:200]}...")
```

### Running Analysis

```python
# Run financial analysis with RAG context
task = "What are the best top 3 ETFs for gold coverage?"
response = agent.run_analysis(task)
print(response)
```

## 📁 Directory Structure

```
financial_documents/
├── research_papers/
│   ├── gold_etf_analysis.pdf
│   ├── market_research.pdf
│   └── portfolio_strategies.pdf
├── company_reports/
│   ├── annual_reports.txt
│   └── quarterly_updates.md
└── market_data/
    ├── historical_prices.csv
    └── volatility_analysis.txt
```

## ⚙️ Configuration Options

### Agent Configuration
- `agent_name`: Name of the agent
- `collection_name`: Qdrant collection name
- `model_name`: LLM model to use
- `max_loops`: Maximum agent execution loops
- `chunk_size`: Text chunk size (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

### Document Processing
- `supported_extensions`: File types to process
- `max_workers`: Concurrent processing threads
- `score_threshold`: Similarity search threshold

## 🔍 Advanced Features

### Custom Embedding Models
```python
# Use different sentence transformer models
from sentence_transformers import SentenceTransformer

custom_model = SentenceTransformer("all-mpnet-base-v2")
# Update the embedding model in QdrantRAGMemory
```

### Cloud Deployment
```python
# Connect to Qdrant cloud
agent = QuantitativeTradingRAGAgent(
    qdrant_url="https://your-instance.qdrant.io",
    qdrant_api_key="your_api_key"
)
```

### Batch Processing
```python
# Process multiple directories
directories = ["./docs1", "./docs2", "./docs3"]
for directory in directories:
    agent.ingest_documents(directory)
```

## 📊 Performance Considerations

- **Chunk Size**: Larger chunks (1000-2000 chars) for detailed analysis, smaller (500-1000) for precise retrieval
- **Overlap**: 10-20% overlap between chunks for better context continuity
- **Concurrency**: Adjust `max_workers` based on your system capabilities
- **Vector Size**: 768 dimensions for sentence-transformers, 1536 for OpenAI embeddings

## 🚨 Error Handling

The system includes comprehensive error handling for:
- File not found errors
- Unsupported file types
- Processing failures
- Network connectivity issues
- Invalid document content

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce chunk size or use cloud Qdrant
   ```python
   agent = QuantitativeTradingRAGAgent(chunk_size=500)
   ```

3. **Processing Failures**: Check file permissions and formats
   ```python
   # Verify supported formats
   processor = DocumentProcessor(supported_extensions=['.pdf', '.txt'])
   ```

### Performance Optimization

- Use SSD storage for document processing
- Increase `max_workers` for multi-core systems
- Consider cloud Qdrant for large document collections
- Implement document caching for frequently accessed files

## 📈 Use Cases

- **Financial Research**: Analyze market reports, earnings calls, and research papers
- **Legal Document Review**: Process contracts, regulations, and case law
- **Academic Research**: Index research papers and academic literature
- **Compliance Monitoring**: Track regulatory changes and compliance requirements
- **Risk Assessment**: Analyze risk reports and market analysis

## 🤝 Contributing

To extend this example:
1. Add support for additional file formats
2. Implement custom embedding strategies
3. Add document versioning and change tracking
4. Integrate with other vector databases
5. Add document summarization capabilities

## 📄 License

This example is part of the Swarms framework and follows the same licensing terms.

## 🆘 Support

For issues and questions:
- Check the Swarms documentation
- Review the example code and error messages
- Ensure all dependencies are properly installed
- Verify Qdrant connection and configuration
