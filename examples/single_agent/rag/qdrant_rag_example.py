"""
Qdrant RAG Example with Document Ingestion

This example demonstrates how to use the agent structure from example.py with Qdrant RAG
to ingest a vast array of PDF documents and text files for advanced quantitative trading analysis.

Features:
- Document ingestion from multiple file types (PDF, TXT, MD)
- Qdrant vector database integration
- Sentence transformer embeddings
- Comprehensive document processing pipeline
- Agent with RAG capabilities for financial analysis
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from swarms import Agent
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.utils.data_to_text import data_to_text


class DocumentProcessor:
    """
    Handles document processing and text extraction from various file formats.

    This class provides functionality to process PDF, TXT, and Markdown files,
    extracting text content for vectorization and storage in the RAG system.
    """

    def __init__(
        self, supported_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the DocumentProcessor.

        Args:
            supported_extensions: List of supported file extensions.
                                Defaults to ['.pdf', '.txt', '.md']
        """
        if supported_extensions is None:
            supported_extensions = [".pdf", ".txt", ".md"]

        self.supported_extensions = supported_extensions

    def process_document(
        self, file_path: Union[str, Path]
    ) -> Optional[Dict[str, str]]:
        """
        Process a single document and extract its text content.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing document metadata and extracted text, or None if processing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None

        if file_path.suffix.lower() not in self.supported_extensions:
            print(f"Unsupported file type: {file_path.suffix}")
            return None

        try:
            # Extract text based on file type
            if file_path.suffix.lower() == ".pdf":
                try:
                    text_content = pdf_to_text(str(file_path))
                except Exception as pdf_error:
                    print(f"Error extracting PDF text: {pdf_error}")
                    # Fallback: try to read as text file
                    with open(
                        file_path,
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        text_content = f.read()
            else:
                try:
                    text_content = data_to_text(str(file_path))
                except Exception as data_error:
                    print(f"Error extracting text: {data_error}")
                    # Fallback: try to read as text file
                    with open(
                        file_path,
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        text_content = f.read()

            # Ensure text_content is a string
            if callable(text_content):
                print(
                    f"Warning: {file_path} returned a callable, trying to call it..."
                )
                try:
                    text_content = text_content()
                except Exception as call_error:
                    print(f"Error calling callable: {call_error}")
                    return None

            if not text_content or not isinstance(text_content, str):
                print(
                    f"No valid text content extracted from: {file_path}"
                )
                return None

            # Clean the text content
            text_content = str(text_content).strip()

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "text_content": text_content,
                "file_size": file_path.stat().st_size,
                "processed_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def process_directory(
        self, directory_path: Union[str, Path], max_workers: int = 4
    ) -> List[Dict[str, str]]:
        """
        Process all supported documents in a directory concurrently.

        Args:
            directory_path: Path to the directory containing documents
            max_workers: Maximum number of concurrent workers for processing

        Returns:
            List of processed document dictionaries
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            print(f"Directory not found: {directory_path}")
            return []

        # Find all supported files
        supported_files = []
        for ext in self.supported_extensions:
            supported_files.extend(directory_path.rglob(f"*{ext}"))
            supported_files.extend(
                directory_path.rglob(f"*{ext.upper()}")
            )

        if not supported_files:
            print(f"No supported files found in: {directory_path}")
            return []

        print(f"Found {len(supported_files)} files to process")

        # Process files concurrently
        processed_documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.process_document, file_path
                ): file_path
                for file_path in supported_files
            }

            for future in concurrent.futures.as_completed(
                future_to_file
            ):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        processed_documents.append(result)
                        print(f"Processed: {result['file_name']}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

        print(
            f"Successfully processed {len(processed_documents)} documents"
        )
        return processed_documents


class QdrantRAGMemory:
    """
    Enhanced Qdrant memory system for RAG operations with document storage.

    This class extends the basic Qdrant memory system to handle document ingestion,
    chunking, and semantic search for large document collections.
    """

    def __init__(
        self,
        collection_name: str = "document_memories",
        vector_size: int = 384,  # Default size for all-MiniLM-L6-v2
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the Qdrant RAG memory system.

        Args:
            collection_name: Name of the Qdrant collection to use
            vector_size: Dimension of the embedding vectors
            url: Optional Qdrant server URL (defaults to local)
            api_key: Optional Qdrant API key for cloud deployment
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize Qdrant client
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(
                ":memory:"
            )  # Local in-memory storage

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Get the actual embedding dimension from the model
        sample_text = "Sample text for dimension check"
        sample_embedding = self.embedding_model.encode(sample_text)
        actual_dimension = len(sample_embedding)

        # Update vector_size to match the actual model dimension
        if actual_dimension != self.vector_size:
            print(
                f"Updating vector size from {self.vector_size} to {actual_dimension} to match model"
            )
            self.vector_size = actual_dimension

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(
            col.name == self.collection_name for col in collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            print(
                f"Created Qdrant collection: {self.collection_name}"
            )

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.

        Args:
            text: Text content to chunk

        Returns:
            List of text chunks
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in ".!?":
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def add_document(
        self, document_data: Dict[str, str]
    ) -> List[str]:
        """
        Add a document to the memory system with chunking.

        Args:
            document_data: Dictionary containing document information

        Returns:
            List of memory IDs for the stored chunks
        """
        text_content = document_data["text_content"]

        # Ensure text_content is a string
        if not isinstance(text_content, str):
            print(
                f"Warning: text_content is not a string: {type(text_content)}"
            )
            text_content = str(text_content)

        chunks = self._chunk_text(text_content)

        memory_ids = []

        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = self.embedding_model.encode(chunk).tolist()

            # Prepare metadata
            metadata = {
                "document_name": document_data["file_name"],
                "document_path": document_data["file_path"],
                "document_type": document_data["file_type"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_text": chunk,
                "timestamp": datetime.utcnow().isoformat(),
                "file_size": document_data["file_size"],
            }

            # Store the chunk
            memory_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=memory_id,
                        payload=metadata,
                        vector=embedding,
                    )
                ],
            )

            memory_ids.append(memory_id)

        print(
            f"Added document '{document_data['file_name']}' with {len(chunks)} chunks"
        )
        return memory_ids

    def add_documents_batch(
        self, documents: List[Dict[str, str]]
    ) -> List[str]:
        """
        Add multiple documents to the memory system.

        Args:
            documents: List of document dictionaries

        Returns:
            List of all memory IDs
        """
        all_memory_ids = []

        for document in documents:
            memory_ids = self.add_document(document)
            all_memory_ids.extend(memory_ids)

        return all_memory_ids

    def add(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a text entry to the memory system (required by Swarms interface).

        Args:
            text: The text content to add
            metadata: Optional metadata for the entry

        Returns:
            str: ID of the stored memory
        """
        if metadata is None:
            metadata = {}

        # Generate embedding for the text
        embedding = self.embedding_model.encode(text).tolist()

        # Prepare metadata
        memory_metadata = {
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "agent_memory",
        }
        memory_metadata.update(metadata)

        # Store the point
        memory_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory_id,
                    payload=memory_metadata,
                    vector=embedding,
                )
            ],
        )

        return memory_id

    def query(
        self,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Query memories based on text similarity.

        Args:
            query_text: The text query to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            include_metadata: Whether to include metadata in results

        Returns:
            List of matching memories with their metadata
        """
        try:
            # Check if collection has any points
            collection_info = self.client.get_collection(
                self.collection_name
            )
            if collection_info.points_count == 0:
                print(
                    "Warning: Collection is empty, no documents to query"
                )
                return []

            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(
                query_text
            ).tolist()

            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )

            memories = []
            for res in results:
                memory = res.payload.copy()
                memory["similarity_score"] = res.score

                if not include_metadata:
                    # Keep only essential information
                    memory = {
                        "chunk_text": memory.get("chunk_text", ""),
                        "document_name": memory.get(
                            "document_name", ""
                        ),
                        "similarity_score": memory[
                            "similarity_score"
                        ],
                    }

                memories.append(memory)

            return memories

        except Exception as e:
            print(f"Error querying collection: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary containing collection statistics
        """
        try:
            collection_info = self.client.get_collection(
                self.collection_name
            )
            return {
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}

    def clear_collection(self):
        """Clear all memories from the collection."""
        self.client.delete_collection(self.collection_name)
        self._create_collection()
        print(f"Cleared collection: {self.collection_name}")


class QuantitativeTradingRAGAgent:
    """
    Advanced quantitative trading agent with RAG capabilities for document analysis.

    This agent combines the structure from example.py with Qdrant RAG to provide
    comprehensive financial analysis based on ingested documents.
    """

    def __init__(
        self,
        agent_name: str = "Quantitative-Trading-RAG-Agent",
        collection_name: str = "financial_documents",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-20250514",
        max_loops: int = 1,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the Quantitative Trading RAG Agent.

        Args:
            agent_name: Name of the agent
            collection_name: Name of the Qdrant collection
            qdrant_url: Optional Qdrant server URL
            qdrant_api_key: Optional Qdrant API key
            model_name: LLM model to use
            max_loops: Maximum number of agent loops
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.agent_name = agent_name
        self.collection_name = collection_name

        # Initialize document processor
        self.document_processor = DocumentProcessor()

        # Initialize Qdrant RAG memory
        self.rag_memory = QdrantRAGMemory(
            collection_name=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize the agent with RAG capabilities
        self.agent = Agent(
            agent_name=agent_name,
            agent_description="Advanced quantitative trading and algorithmic analysis agent with RAG capabilities",
            system_prompt="""You are an expert quantitative trading agent with deep expertise in:
            - Algorithmic trading strategies and implementation
            - Statistical arbitrage and market making
            - Risk management and portfolio optimization
            - High-frequency trading systems
            - Market microstructure analysis
            - Quantitative research methodologies
            - Financial mathematics and stochastic processes
            - Machine learning applications in trading
            
            Your core responsibilities include:
            1. Developing and backtesting trading strategies
            2. Analyzing market data and identifying alpha opportunities
            3. Implementing risk management frameworks
            4. Optimizing portfolio allocations
            5. Conducting quantitative research
            6. Monitoring market microstructure
            7. Evaluating trading system performance
            
            You have access to a comprehensive document database through RAG (Retrieval-Augmented Generation).
            When answering questions, you can search through this database to find relevant information
            and provide evidence-based responses.
            
            You maintain strict adherence to:
            - Mathematical rigor in all analyses
            - Statistical significance in strategy development
            - Risk-adjusted return optimization
            - Market impact minimization
            - Regulatory compliance
            - Transaction cost analysis
            - Performance attribution
            
            You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
            model_name=model_name,
            dynamic_temperature_enabled=True,
            output_type="str-all-except-first",
            max_loops=max_loops,
            dynamic_context_window=True,
            long_term_memory=self.rag_memory,
        )

    def ingest_documents(
        self, documents_path: Union[str, Path]
    ) -> int:
        """
        Ingest documents from a directory into the RAG system.

        Args:
            documents_path: Path to directory containing documents

        Returns:
            Number of documents successfully ingested
        """
        print(f"Starting document ingestion from: {documents_path}")

        try:
            # Process documents
            processed_documents = (
                self.document_processor.process_directory(
                    documents_path
                )
            )

            if not processed_documents:
                print("No documents to ingest")
                return 0

            # Add documents to RAG memory
            memory_ids = self.rag_memory.add_documents_batch(
                processed_documents
            )

            print(
                f"Successfully ingested {len(processed_documents)} documents"
            )
            print(f"Created {len(memory_ids)} memory chunks")

            return len(processed_documents)

        except Exception as e:
            print(f"Error during document ingestion: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def query_documents(
        self, query: str, limit: int = 5
    ) -> List[Dict]:
        """
        Query the document database for relevant information.

        Args:
            query: The query text
            limit: Maximum number of results to return

        Returns:
            List of relevant document chunks
        """
        return self.rag_memory.query(query, limit=limit)

    def run_analysis(self, task: str) -> str:
        """
        Run a financial analysis task using the agent with RAG capabilities.

        Args:
            task: The analysis task to perform

        Returns:
            Agent's response to the task
        """
        print(f"Running analysis task: {task}")

        # First, query the document database for relevant context
        relevant_docs = self.query_documents(task, limit=3)

        if relevant_docs:
            # Enhance the task with relevant document context
            context = "\n\nRelevant Document Information:\n"
            for i, doc in enumerate(relevant_docs, 1):
                context += f"\nDocument {i}: {doc.get('document_name', 'Unknown')}\n"
                context += f"Relevance Score: {doc.get('similarity_score', 0):.3f}\n"
                context += (
                    f"Content: {doc.get('chunk_text', '')[:500]}...\n"
                )

            enhanced_task = f"{task}\n\n{context}"
        else:
            enhanced_task = task

        # Run the agent
        response = self.agent.run(enhanced_task)
        return response

    def get_database_stats(self) -> Dict:
        """
        Get statistics about the document database.

        Returns:
            Dictionary containing database statistics
        """
        return self.rag_memory.get_collection_stats()


def main():
    """
    Main function demonstrating the Qdrant RAG agent with document ingestion.
    """
    from datetime import datetime

    # Example usage
    print("🚀 Initializing Quantitative Trading RAG Agent...")

    # Initialize the agent (you can set environment variables for Qdrant cloud)
    agent = QuantitativeTradingRAGAgent(
        agent_name="Quantitative-Trading-RAG-Agent",
        collection_name="financial_documents",
        qdrant_url=os.getenv(
            "QDRANT_URL"
        ),  # Optional: For cloud deployment
        qdrant_api_key=os.getenv(
            "QDRANT_API_KEY"
        ),  # Optional: For cloud deployment
        model_name="claude-sonnet-4-20250514",
        max_loops=1,
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Example: Ingest documents from a directory
    documents_path = "documents"  # Path to your documents
    if os.path.exists(documents_path):
        print(f"Found documents directory: {documents_path}")
        try:
            agent.ingest_documents(documents_path)
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            print("Continuing without document ingestion...")
    else:
        print(f"Documents directory not found: {documents_path}")
        print("Creating a sample document for demonstration...")

        # Create a sample document
        try:
            sample_doc = {
                "file_path": "sample_financial_analysis.txt",
                "file_name": "sample_financial_analysis.txt",
                "file_type": ".txt",
                "text_content": """
                Gold ETFs: A Comprehensive Investment Guide
                
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
                
                Market analysis shows that gold has historically served as a store of value
                and hedge against inflation. Recent market conditions have increased interest
                in gold investments due to economic uncertainty and geopolitical tensions.
                """,
                "file_size": 1024,
                "processed_at": datetime.utcnow().isoformat(),
            }

            # Add the sample document to the RAG memory
            memory_ids = agent.rag_memory.add_document(sample_doc)
            print(
                f"Added sample document with {len(memory_ids)} chunks"
            )

        except Exception as e:
            print(f"Error creating sample document: {e}")
            print("Continuing without sample document...")

    # Example: Query the database
    print("\n📊 Querying document database...")
    try:
        query_results = agent.query_documents(
            "gold ETFs investment strategies", limit=3
        )
        print(f"Found {len(query_results)} relevant document chunks")

        if query_results:
            print("Sample results:")
            for i, result in enumerate(query_results[:2], 1):
                print(
                    f"  {i}. {result.get('document_name', 'Unknown')} (Score: {result.get('similarity_score', 0):.3f})"
                )
        else:
            print(
                "No documents found in database. This is expected if no documents were ingested."
            )
    except Exception as e:
        print(f"❌ Query failed: {e}")

    # Example: Run financial analysis
    print("\n💹 Running financial analysis...")
    analysis_task = "What are the best top 3 ETFs for gold coverage and what are their key characteristics?"
    try:
        response = agent.run_analysis(analysis_task)
        print("\n📈 Analysis Results:")
        print(response)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("This might be due to API key or model access issues.")
        print("Continuing with database statistics...")

        # Try a simpler query that doesn't require the LLM
        print("\n🔍 Trying simple document query instead...")
        try:
            simple_results = agent.query_documents(
                "what do you see in the document?", limit=2
            )
            if simple_results:
                print("Simple query results:")
                for i, result in enumerate(simple_results, 1):
                    print(
                        f"  {i}. {result.get('document_name', 'Unknown')}"
                    )
                    print(
                        f"     Content preview: {result.get('chunk_text', '')[:100]}..."
                    )
            else:
                print("No results from simple query")
        except Exception as simple_error:
            print(f"Simple query also failed: {simple_error}")

    # Get database statistics
    print("\n📊 Database Statistics:")
    try:
        stats = agent.get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ Failed to get database statistics: {e}")

    print("\n✅ Example completed successfully!")
    print("💡 To test with your own documents:")
    print("   1. Create a 'documents' directory")
    print("   2. Add PDF, TXT, or MD files")
    print("   3. Run the script again")


if __name__ == "__main__":
    main()
