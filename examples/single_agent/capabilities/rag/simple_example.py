"""
Simple Example: Qdrant RAG with Document Ingestion

This is a simplified example showing the basic usage of the Qdrant RAG system
for document ingestion and querying.
"""

from pathlib import Path
from examples.single_agent.rag.qdrant_rag_example import (
    QuantitativeTradingRAGAgent,
)


def create_sample_documents():
    """
    Create sample documents for demonstration purposes.
    """
    # Create a sample documents directory
    docs_dir = Path("./sample_documents")
    docs_dir.mkdir(exist_ok=True)

    # Create sample text files
    sample_texts = {
        "gold_etf_guide.txt": """
        Gold ETFs: A Comprehensive Guide
        
        Gold ETFs (Exchange-Traded Funds) provide investors with exposure to gold prices
        without the need to physically store the precious metal. These funds track the
        price of gold and offer several advantages including liquidity, diversification,
        and ease of trading.
        
        Top Gold ETFs include:
        1. SPDR Gold Shares (GLD) - Largest gold ETF with high liquidity
        2. iShares Gold Trust (IAU) - Lower expense ratio alternative
        3. Aberdeen Standard Physical Gold ETF (SGOL) - Swiss storage option
        
        Investment strategies for gold ETFs:
        - Portfolio diversification (5-10% allocation)
        - Inflation hedge
        - Safe haven during market volatility
        - Tactical trading opportunities
        """,
        "market_analysis.txt": """
        Market Analysis: Gold Investment Trends
        
        Gold has historically served as a store of value and hedge against inflation.
        Recent market conditions have increased interest in gold investments due to:
        
        - Economic uncertainty and geopolitical tensions
        - Inflation concerns and currency devaluation
        - Central bank policies and interest rate environment
        - Portfolio diversification needs
        
        Key factors affecting gold prices:
        - US Dollar strength/weakness
        - Real interest rates
        - Central bank gold purchases
        - Market risk sentiment
        - Supply and demand dynamics
        
        Investment recommendations:
        - Consider gold as 5-15% of total portfolio
        - Use dollar-cost averaging for entry
        - Monitor macroeconomic indicators
        - Rebalance periodically
        """,
        "portfolio_strategies.txt": """
        Portfolio Strategies: Incorporating Gold
        
        Strategic allocation to gold can enhance portfolio performance through:
        
        1. Risk Reduction:
           - Negative correlation with equities during crises
           - Volatility dampening effects
           - Drawdown protection
        
        2. Return Enhancement:
           - Long-term appreciation potential
           - Inflation-adjusted returns
           - Currency diversification benefits
        
        3. Implementation Methods:
           - Physical gold (coins, bars)
           - Gold ETFs and mutual funds
           - Gold mining stocks
           - Gold futures and options
        
        Optimal allocation ranges:
        - Conservative: 5-10%
        - Moderate: 10-15%
        - Aggressive: 15-20%
        
        Rebalancing frequency: Quarterly to annually
        """,
    }

    # Write sample files
    for filename, content in sample_texts.items():
        file_path = docs_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

    print(
        f"Created {len(sample_texts)} sample documents in {docs_dir}"
    )
    return docs_dir


def main():
    """
    Main function demonstrating basic Qdrant RAG usage.
    """
    print("🚀 Qdrant RAG Simple Example")
    print("=" * 50)

    # Create sample documents
    docs_dir = create_sample_documents()

    # Initialize the RAG agent
    print("\n📊 Initializing Quantitative Trading RAG Agent...")
    agent = QuantitativeTradingRAGAgent(
        agent_name="Simple-Financial-Agent",
        collection_name="sample_financial_docs",
        model_name="claude-sonnet-4-20250514",
        chunk_size=800,  # Smaller chunks for sample documents
        chunk_overlap=100,
    )

    # Ingest the sample documents
    print(f"\n📚 Ingesting documents from {docs_dir}...")
    num_ingested = agent.ingest_documents(docs_dir)
    print(f"✅ Successfully ingested {num_ingested} documents")

    # Query the document database
    print("\n🔍 Querying document database...")
    queries = [
        "What are the top gold ETFs?",
        "How should I allocate gold in my portfolio?",
        "What factors affect gold prices?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = agent.query_documents(query, limit=2)
        print(f"Found {len(results)} relevant chunks:")

        for i, result in enumerate(results, 1):
            print(
                f"  {i}. {result['document_name']} (Score: {result['similarity_score']:.3f})"
            )
            print(f"     Content: {result['chunk_text'][:150]}...")

    # Run a comprehensive analysis
    print("\n💹 Running comprehensive analysis...")
    analysis_task = "Based on the available documents, provide a summary of gold ETF investment strategies and portfolio allocation recommendations."

    try:
        response = agent.run_analysis(analysis_task)
        print("\n📈 Analysis Results:")
        print("-" * 30)
        print(response)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("This might be due to API key or model access issues.")

    # Show database statistics
    print("\n📊 Database Statistics:")
    stats = agent.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup
    print("\n🧹 Cleaning up sample documents...")
    import shutil

    if docs_dir.exists():
        shutil.rmtree(docs_dir)
        print("Sample documents removed.")

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
