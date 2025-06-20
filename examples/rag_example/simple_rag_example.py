"""
Simple RAG Example with Swarms Framework

A concise example showing how to use the RAG integration with Swarms Agent.
This example demonstrates the core RAG functionality in a simple, easy-to-understand way.
"""

import time
from swarms.structs import Agent, RAGConfig


class SimpleMemoryStore:
    """Simple in-memory memory store for demonstration"""
    
    def __init__(self):
        self.memories = []
    
    def add(self, content: str, metadata: dict = None) -> bool:
        """Add content to memory"""
        self.memories.append({
            'content': content,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        return True
    
    def query(self, query: str, top_k: int = 3, similarity_threshold: float = 0.5) -> list:
        """Simple keyword-based query"""
        query_lower = query.lower()
        results = []
        
        for memory in self.memories:
            content_lower = memory['content'].lower()
            # Simple relevance score
            relevance = sum(1 for word in query_lower.split() if word in content_lower)
            relevance = min(relevance / len(query_lower.split()), 1.0)
            
            if relevance >= similarity_threshold:
                results.append({
                    'content': memory['content'],
                    'score': relevance,
                    'metadata': memory['metadata']
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]


def main():
    """Main example demonstrating RAG functionality"""
    print("🚀 Simple RAG Example with Swarms Framework")
    print("=" * 50)
    
    # 1. Initialize memory store
    print("\n1. Setting up memory store...")
    memory_store = SimpleMemoryStore()
    
    # Add some knowledge to memory
    knowledge_items = [
        "Python is a versatile programming language used for web development, data science, and AI.",
        "Machine learning models learn patterns from data to make predictions.",
        "The Swarms framework enables building sophisticated multi-agent systems.",
        "RAG (Retrieval-Augmented Generation) enhances AI responses with external knowledge.",
        "Vector databases store embeddings for efficient similarity search."
    ]
    
    for item in knowledge_items:
        memory_store.add(item, {'source': 'knowledge_base'})
    
    print(f"✅ Added {len(knowledge_items)} knowledge items to memory")
    
    # 2. Configure RAG
    print("\n2. Configuring RAG...")
    rag_config = RAGConfig(
        similarity_threshold=0.3,  # Lower threshold for demo
        max_results=2,
        auto_save_to_memory=True,
        query_every_loop=False,  # Disable to avoid issues
        enable_conversation_summaries=True
    )
    
    # 3. Create agent with RAG - using built-in model handling
    agent = Agent(
        model_name="gpt-4o-mini",  # Direct model specification
        temperature=0.7,
        max_tokens=300,
        agent_name="RAG-Demo-Agent",
        long_term_memory=memory_store,
        rag_config=rag_config,
        max_loops=1,  # Reduce loops to avoid issues
        verbose=True
    )
    
    print(f"✅ Agent created with RAG enabled: {agent.is_rag_enabled()}")
    
    # 4. Test RAG functionality
    print("\n4. Testing RAG functionality...")
    
    test_queries = [
        "What is Python used for?",
        "How do machine learning models work?",
        "What is the Swarms framework?",
        "Explain RAG systems"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        try:
            # Run the agent
            response = agent.run(query)
            print(f"🤖 Response: {response}")
            
            # Check RAG stats
            stats = agent.get_rag_stats()
            print(f"📊 RAG Stats: {stats.get('loops_processed', 0)} loops processed")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)
    

    
    try:
        # Save custom content
        success = agent.save_to_rag_memory(
            "Custom knowledge: The agent successfully used RAG to enhance responses.",
            {'source': 'manual_test'}
        )
        print(f"💾 Manual save: {success}")
        
        # Query memory directly
        result = agent.query_rag_memory("What is custom knowledge?")
        print(f"🔍 Direct query result: {result[:100]}...")
        
        # Search memories
        search_results = agent.search_memories("Python", top_k=2)
        print(f"🔎 Search results: {len(search_results)} items found")
        
    except Exception as e:
        print(f"❌ Error in manual operations: {e}")
    
    # 6. Final statistics
    print("\n6. Final RAG statistics...")
    try:
        final_stats = agent.get_rag_stats()
        print(f"📈 Final Stats: {final_stats}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    print("\n🎉 RAG example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main() 