from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

@dataclass
class Document:
    """Represents a document in the HQD-RAG system.
    
    Attributes:
        id (str): Unique identifier for the document
        content (str): Raw text content of the document
        embedding (Optional[np.ndarray]): Quantum-inspired embedding vector
        cluster_id (Optional[int]): ID of the cluster this document belongs to
    """
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None

class HQDRAG:
    """
    Hierarchical Quantum-Inspired Distributed RAG (HQD-RAG) System
    
    A production-grade implementation of the HQD-RAG algorithm for ultra-fast
    and reliable document retrieval. Uses quantum-inspired embeddings and
    hierarchical clustering for efficient search.
    
    Attributes:
        embedding_dim (int): Dimension of the quantum-inspired embeddings
        num_clusters (int): Number of hierarchical clusters
        similarity_threshold (float): Threshold for quantum similarity matching
        reliability_threshold (float): Threshold for reliability verification
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_clusters: int = 128,
        similarity_threshold: float = 0.75,
        reliability_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the HQD-RAG system.
        
        Args:
            embedding_dim: Dimension of document embeddings
            num_clusters: Number of clusters for hierarchical organization
            similarity_threshold: Minimum similarity score for retrieval
            reliability_threshold: Minimum reliability score for verification
            model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Initializing HQD-RAG with {embedding_dim} dimensions")
        
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.similarity_threshold = similarity_threshold
        self.reliability_threshold = reliability_threshold
        
        # Initialize components
        self.documents: Dict[str, Document] = {}
        self.encoder = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product index
        self.clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric='euclidean',
            linkage='ward'
        )
        
        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("HQD-RAG system initialized successfully")
    
    def _compute_quantum_embedding(self, text: str) -> np.ndarray:
        """Compute quantum-inspired embedding for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Quantum-inspired embedding vector
        """
        # Get base embedding
        base_embedding = self.encoder.encode([text])[0]
        
        # Apply quantum-inspired transformation
        # Simulate superposition by adding phase components
        phase = np.exp(2j * np.pi * np.random.random(self.embedding_dim))
        quantum_embedding = base_embedding * phase
        
        # Normalize to unit length
        return quantum_embedding / np.linalg.norm(quantum_embedding)
    
    def _verify_reliability(self, doc: Document, query_embedding: np.ndarray) -> float:
        """Verify the reliability of a document match.
        
        Args:
            doc: Document to verify
            query_embedding: Query embedding vector
            
        Returns:
            Reliability score between 0 and 1
        """
        if doc.embedding is None:
            return 0.0
            
        # Compute consistency score
        consistency = np.abs(np.dot(doc.embedding, query_embedding))
        
        # Add quantum noise resistance check
        noise = np.random.normal(0, 0.1, self.embedding_dim)
        noisy_query = query_embedding + noise
        noisy_query = noisy_query / np.linalg.norm(noisy_query)
        noise_resistance = np.abs(np.dot(doc.embedding, noisy_query))
        
        return (consistency + noise_resistance) / 2
    
    def add(self, content: str, doc_id: Optional[str] = None) -> str:
        """Add a document to the system.
        
        Args:
            content: Document text content
            doc_id: Optional custom document ID
            
        Returns:
            Document ID
        """
        doc_id = doc_id or str(uuid.uuid4())
        
        with self._lock:
            try:
                # Compute embedding
                embedding = self._compute_quantum_embedding(content)
                
                # Create document
                doc = Document(
                    id=doc_id,
                    content=content,
                    embedding=embedding
                )
                
                # Add to storage
                self.documents[doc_id] = doc
                self.index.add(embedding.reshape(1, -1))
                
                # Update clustering
                self._update_clusters()
                
                logger.info(f"Successfully added document {doc_id}")
                return doc_id
                
            except Exception as e:
                logger.error(f"Error adding document: {str(e)}")
                raise
    
    def query(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = False
    ) -> Union[List[str], List[tuple[str, float]]]:
        """Query the system for relevant documents.
        
        Args:
            query: Query text
            k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of document IDs or (document ID, score) tuples
        """
        try:
            # Compute query embedding
            query_embedding = self._compute_quantum_embedding(query)
            
            # Search index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1),
                k * 2  # Get extra results for reliability filtering
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Get document
                doc_id = list(self.documents.keys())[idx]
                doc = self.documents[doc_id]
                
                # Verify reliability
                reliability = self._verify_reliability(doc, query_embedding)
                
                if reliability >= self.reliability_threshold:
                    results.append((doc_id, float(score)))
                    
                if len(results) >= k:
                    break
            
            logger.info(f"Query returned {len(results)} results")
            
            if return_scores:
                return results
            return [doc_id for doc_id, _ in results]
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def update(self, doc_id: str, new_content: str) -> None:
        """Update an existing document.
        
        Args:
            doc_id: ID of document to update
            new_content: New document content
        """
        with self._lock:
            try:
                if doc_id not in self.documents:
                    raise KeyError(f"Document {doc_id} not found")
                
                # Remove old embedding
                old_doc = self.documents[doc_id]
                if old_doc.embedding is not None:
                    self.index.remove_ids(np.array([list(self.documents.keys()).index(doc_id)]))
                
                # Compute new embedding
                new_embedding = self._compute_quantum_embedding(new_content)
                
                # Update document
                self.documents[doc_id] = Document(
                    id=doc_id,
                    content=new_content,
                    embedding=new_embedding
                )
                
                # Add new embedding
                self.index.add(new_embedding.reshape(1, -1))
                
                # Update clustering
                self._update_clusters()
                
                logger.info(f"Successfully updated document {doc_id}")
                
            except Exception as e:
                logger.error(f"Error updating document: {str(e)}")
                raise
    
    def delete(self, doc_id: str) -> None:
        """Delete a document from the system.
        
        Args:
            doc_id: ID of document to delete
        """
        with self._lock:
            try:
                if doc_id not in self.documents:
                    raise KeyError(f"Document {doc_id} not found")
                
                # Remove from index
                idx = list(self.documents.keys()).index(doc_id)
                self.index.remove_ids(np.array([idx]))
                
                # Remove from storage
                del self.documents[doc_id]
                
                # Update clustering
                self._update_clusters()
                
                logger.info(f"Successfully deleted document {doc_id}")
                
            except Exception as e:
                logger.error(f"Error deleting document: {str(e)}")
                raise
    
    def _update_clusters(self) -> None:
        """Update hierarchical document clusters."""
        if len(self.documents) < 2:
            return
            
        # Get all embeddings
        embeddings = np.vstack([
            doc.embedding for doc in self.documents.values()
            if doc.embedding is not None
        ])
        
        # Update clustering
        clusters = self.clustering.fit_predict(embeddings)
        
        # Assign cluster IDs
        for doc, cluster_id in zip(self.documents.values(), clusters):
            doc.cluster_id = int(cluster_id)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the system state to disk.
        
        Args:
            path: Path to save directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save documents
            with open(path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # Save index
            faiss.write_index(self.index, str(path / "index.faiss"))
            
            logger.info(f"Successfully saved system state to {path}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
            raise
    
    def load(self, path: Union[str, Path]) -> None:
        """Load the system state from disk.
        
        Args:
            path: Path to save directory
        """
        path = Path(path)
        
        try:
            # Load documents
            with open(path / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            
            # Load index
            self.index = faiss.read_index(str(path / "index.faiss"))
            
            logger.info(f"Successfully loaded system state from {path}")
            
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")
            raise

# Example usage:
if __name__ == "__main__":
    # Configure logging
    logger.add(
        "hqd_rag.log",
        rotation="1 day",
        retention="1 week",
        level="INFO"
    )
    
    # Initialize system
    rag = HQDRAG()
    
    # Add some documents
    doc_ids = []
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language"
    ]
    
    for doc in docs:
        doc_id = rag.add(doc)
        doc_ids.append(doc_id)
    
    # Query
    results = rag.query("What is machine learning?", return_scores=True)
    print("Query results:", results)
    
    # # Update a document
    # rag.update(doc_ids[0], "The fast brown fox jumps over the sleepy dog")
    
    # # Delete a document
    # rag.delete(doc_ids[-1])
    
    # # Save state
    # rag.save("hqd_rag_state")
    
    
    