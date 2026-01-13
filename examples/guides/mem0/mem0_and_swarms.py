import os
from typing import List, Dict, Any, Optional

from mem0 import MemoryClient
from swarms import Agent
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.utils.data_to_text import csv_to_text
from dotenv import load_dotenv

load_dotenv()

# Initialize Mem0 memory client as singleton at module level
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
memory_client = (
    MemoryClient(api_key=MEM0_API_KEY) if MEM0_API_KEY else None
)


def chunk_text(
    text: str, chunk_size: int = 2000, overlap: int = 200
) -> List[str]:
    """
    Split text into chunks with overlap for better context preservation.

    Args:
        text (str): Text to chunk.
        chunk_size (int): Size of each chunk in characters. Defaults to 2000.
        overlap (int): Number of characters to overlap between chunks.
            Defaults to 200.

    Returns:
        List[str]: List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def read_pdf(pdf_path: str) -> str:
    """
    Read and extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content from the PDF.

    Raises:
        FileNotFoundError: If the PDF file is not found.
        Exception: If there is an error reading the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Reading PDF file: {pdf_path}")

    try:
        text = pdf_to_text(pdf_path)
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")


def read_csv_file(csv_path: str) -> str:
    """
    Read and convert CSV file content to text format.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        str: Text representation of the CSV content.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        Exception: If there is an error reading the CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        text = csv_to_text(csv_path)
        return text
    except Exception as e:
        raise Exception(f"Error reading CSV {csv_path}: {str(e)}")


def process_documents(
    pdf_paths: Optional[List[str]] = None,
    csv_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Process multiple PDF and CSV files and extract their content.

    Args:
        pdf_paths (Optional[List[str]]): List of paths to PDF files.
        csv_paths (Optional[List[str]]): List of paths to CSV files.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing document
            information with keys: 'type', 'path', 'content', 'filename'.
    """
    documents = []

    if pdf_paths:
        for pdf_path in pdf_paths:
            try:
                content = read_pdf(pdf_path)
                filename = os.path.basename(pdf_path)
                documents.append(
                    {
                        "type": "pdf",
                        "path": pdf_path,
                        "content": content,
                        "filename": filename,
                    }
                )
            except Exception as e:
                print(
                    f"Warning: Failed to process PDF {pdf_path}: {e}"
                )

    if csv_paths:
        for csv_path in csv_paths:
            try:
                content = read_csv_file(csv_path)
                filename = os.path.basename(csv_path)
                documents.append(
                    {
                        "type": "csv",
                        "path": csv_path,
                        "content": content,
                        "filename": filename,
                    }
                )
            except Exception as e:
                print(
                    f"Warning: Failed to process CSV {csv_path}: {e}"
                )

    return documents


def add_documents_to_mem0(
    documents: List[Dict[str, Any]],
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 2000,
) -> List[str]:
    """
    Add processed documents to Mem0 memory by chunking and storing content.

    Args:
        documents (List[Dict[str, Any]]): List of document dictionaries
            from process_documents function.
        user_id (str): User identifier for scoping memories.
        metadata (Optional[Dict[str, Any]]): Optional metadata to attach
            to memories.
        chunk_size (int): Size of chunks for large documents. Defaults to 2000.

    Returns:
        List[str]: List of memory IDs created.
    """
    memory_ids = []

    for doc in documents:
        try:
            content = doc["content"]
            filename = doc["filename"]
            doc_type = doc["type"]

            # Chunk large documents
            chunks = chunk_text(content, chunk_size=chunk_size)

            print(
                f"Adding {len(chunks)} chunks from {filename} to memory..."
            )

            for chunk_idx, chunk in enumerate(chunks):
                # Prepare metadata with document info
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata.update(
                    {
                        "document_type": doc_type,
                        "filename": filename,
                        "file_path": doc["path"],
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )

                # Store chunk as a memory with the actual content
                # Use infer=False to store raw content, not inferred memories
                messages = [
                    {
                        "role": "user",
                        "content": f"Document chunk {chunk_idx + 1}/{len(chunks)} "
                        f"from {filename} ({doc_type.upper()}):\n\n{chunk}",
                    },
                    {
                        "role": "assistant",
                        "content": f"This is chunk {chunk_idx + 1} of {len(chunks)} "
                        f"from the document {filename}. "
                        f"Content: {chunk}",
                    },
                ]

                try:
                    result = memory_client.add(
                        messages=messages,
                        user_id=user_id,
                        metadata=doc_metadata,
                        infer=False,  # Store raw content, don't infer
                    )

                    # Extract memory IDs from result
                    if isinstance(result, dict):
                        if "id" in result:
                            memory_ids.append(result["id"])
                        elif "ids" in result:
                            memory_ids.extend(result["ids"])
                        elif "results" in result:
                            # Handle platform response format
                            for res in result.get("results", []):
                                if "id" in res:
                                    memory_ids.append(res["id"])
                    elif isinstance(result, list):
                        memory_ids.extend(
                            [
                                r["id"]
                                for r in result
                                if isinstance(r, dict) and "id" in r
                            ]
                        )

                except Exception as e:
                    print(
                        f"Warning: Failed to add chunk {chunk_idx + 1} "
                        f"from {filename}: {e}"
                    )

        except Exception as e:
            print(
                f"Warning: Failed to add document {doc.get('filename', 'unknown')} "
                f"to memory: {e}"
            )

    print(f"Successfully added {len(memory_ids)} memory chunks")
    return memory_ids


def search_mem0_memory(
    query: str,
    user_id: str,
    limit: int = 5,
) -> List[Any]:
    """
    Search Mem0 memory for relevant information.

    Args:
        query (str): Search query.
        user_id (str): User identifier for scoping search.
        limit (int): Maximum number of results to return. Defaults to 5.

    Returns:
        List[Any]: List of relevant memories.
    """
    if memory_client is None:
        print("Error: Memory client is not initialized")
        return []

    if not user_id or not user_id.strip():
        print("Error: user_id is required and cannot be empty")
        return []

    # Try different filter formats for Mem0 Platform API
    # Format 1: Simple user_id in filters
    try:
        filters = {"user_id": user_id}
        results = memory_client.search(
            query=query, filters=filters, limit=limit
        )
        if results is not None:
            if not isinstance(results, list):
                return [results]
            return results
    except Exception as e1:
        print(f"Search attempt 1 failed: {e1}")

        # Format 2: AND logic wrapper
        try:
            filters = {"AND": [{"user_id": user_id}]}
            results = memory_client.search(
                query=query, filters=filters, limit=limit
            )
            if results is not None:
                if not isinstance(results, list):
                    return [results]
                return results
        except Exception as e2:
            print(f"Search attempt 2 failed: {e2}")

            # Format 3: Try with user_id as direct parameter (if supported)
            try:
                # Some APIs accept user_id as a direct parameter
                results = memory_client.search(
                    query=query, user_id=user_id, limit=limit
                )
                if results is not None:
                    if not isinstance(results, list):
                        return [results]
                    return results
            except Exception as e3:
                print(f"All search formats failed. Last error: {e3}")
                import traceback

                traceback.print_exc()
                return []

    return []


def create_rag_context(
    query: str,
    user_id: str,
    limit: int = 5,
) -> str:
    """
    Create RAG context by searching Mem0 memory.

    Args:
        query (str): User query.
        user_id (str): User identifier.
        limit (int): Maximum number of relevant memories to retrieve.

    Returns:
        str: Formatted context string from retrieved memories.
    """
    results = search_mem0_memory(
        query=query, user_id=user_id, limit=limit
    )

    if not results:
        return "No relevant information found in memory."

    context_parts = []
    for i, result in enumerate(results, 1):
        # Handle different result formats
        if isinstance(result, str):
            content = result
            filename = f"Memory {i}"
        elif isinstance(result, dict):
            # Extract content from various possible keys
            content = (
                result.get("memory")
                or result.get("content")
                or result.get("text")
                or result.get("message")
                or str(result)
            )
            # Extract metadata
            metadata = result.get("metadata", {})
            if isinstance(metadata, dict):
                filename = metadata.get("filename", f"Memory {i}")
            else:
                filename = f"Memory {i}"
        else:
            content = str(result)
            filename = f"Memory {i}"

        context_parts.append(
            f"[Document {i}: {filename}]\n{content}\n"
        )

    return "\n".join(context_parts)


def create_rag_agent(
    model_name: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    verbose: bool = True,
) -> Agent:
    """
    Create a Swarms Agent configured for RAG operations.

    Args:
        model_name (str): Name of the LLM model to use.
            Defaults to "gpt-4o-mini".
        system_prompt (Optional[str]): Custom system prompt.
            If None, uses default RAG-focused prompt.
        max_loops (int): Maximum number of agent loops. Defaults to 1.
        temperature (float): Model temperature. Defaults to 0.7.
        verbose (bool): Enable verbose output. Defaults to True.

    Returns:
        Agent: Configured Swarms Agent instance.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant with access to a knowledge base. "
            "Use the provided context from documents to answer questions "
            "accurately. If the context doesn't contain enough information, "
            "say so clearly. Always cite which document your information "
            "comes from when possible."
        )

    agent = Agent(
        agent_name="RAG-Agent",
        agent_description="RAG-enabled agent with Mem0 memory",
        model_name=model_name,
        system_prompt=system_prompt,
        max_loops="auto",
        temperature=temperature,
        verbose=verbose,
        interactive=True,
    )

    return agent


def run_rag_query(
    query: str,
    agent: Agent,
    user_id: str,
    add_to_memory: bool = True,
    context_limit: int = 5,
) -> str:
    """
    Run a RAG query using the agent with Mem0 memory context.

    Args:
        query (str): User query/question.
        agent (Agent): Swarms Agent instance.
        user_id (str): User identifier.
        add_to_memory (bool): Whether to store the interaction in Mem0.
            Defaults to True.
        context_limit (int): Maximum number of memories to retrieve.
            Defaults to 5.

    Returns:
        str: Agent's response to the query.
    """
    # Get RAG context from Mem0
    context = create_rag_context(
        query=query, user_id=user_id, limit=context_limit
    )

    # Construct prompt with context
    prompt = f"""Context from knowledge base:
{context}

User Question: {query}

Please answer the question using the context provided above. 
If the context doesn't contain enough information, say so clearly."""

    # Run agent
    response = agent.run(prompt)

    # Store interaction in Mem0 if requested
    if add_to_memory:
        try:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            memory_client.add(messages=messages, user_id=user_id)
        except Exception as e:
            print(
                f"Warning: Failed to add interaction to memory: {e}"
            )

    return response


def setup_rag_agent_with_mem0(
    pdf_paths: Optional[List[str]] = None,
    csv_paths: Optional[List[str]] = None,
    user_id: str = "default_user",
    model_name: str = "gpt-4o-mini",
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 2000,
) -> Dict[str, Any]:
    """
    Complete setup function to create a RAG agent with Mem0 integration.

    Args:
        pdf_paths (Optional[List[str]]): List of PDF file paths to process.
        csv_paths (Optional[List[str]]): List of CSV file paths to process.
        user_id (str): User identifier for Mem0. Defaults to "default_user".
        model_name (str): LLM model name. Defaults to "gpt-4o-mini".
        metadata (Optional[Dict[str, Any]]): Additional metadata for documents.
        chunk_size (int): Size of chunks for large documents. Defaults to 2000.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'agent': Swarms Agent instance
            - 'user_id': User identifier
            - 'documents': Processed documents
            - 'memory_ids': List of created memory IDs
    """
    # Process documents
    documents = process_documents(
        pdf_paths=pdf_paths, csv_paths=csv_paths
    )

    if not documents:
        print("Warning: No documents were successfully processed.")

    # Add documents to Mem0
    memory_ids = add_documents_to_mem0(
        documents=documents,
        user_id=user_id,
        metadata=metadata,
        chunk_size=chunk_size,
    )

    # Create agent
    agent = create_rag_agent(model_name=model_name)

    return {
        "agent": agent,
        "user_id": user_id,
        "documents": documents,
        "memory_ids": memory_ids,
    }


# Example usage function
def example_usage():
    """
    Example usage of the RAG agent with Mem0.

    This function demonstrates how to use the RAG agent system.
    """
    # Example: Setup with documents
    setup = setup_rag_agent_with_mem0(
        pdf_paths=["pdf.pdf"],
        user_id="example_user",
        model_name="gpt-4.1",
        chunk_size=2000,
    )

    agent = setup["agent"]
    user_id = setup["user_id"]

    # Run queries
    response1 = run_rag_query(
        query=None,
        agent=agent,
        user_id=user_id,
    )
    print(f"Response: {response1}\n")


if __name__ == "__main__":
    # Run example if executed directly
    example_usage()
