from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Union

from doc_master import doc_master
from tenacity import retry, stop_after_attempt, wait_exponential

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="add_docs_to_agents")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def _process_document(doc_path: Union[str, Path]) -> str:
    """Safely process a single document with retries.

    Args:
        doc_path: Path to the document to process

    Returns:
        Processed document text

    Raises:
        Exception: If document processing fails after retries
    """
    try:
        return doc_master(
            file_path=str(doc_path), output_type="string"
        )
    except Exception as e:
        logger.error(
            f"Error processing document {doc_path}: {str(e)}"
        )
        raise


def handle_input_docs(
    agents: Any,
    docs: Optional[List[Union[str, Path]]] = None,
    doc_folder: Optional[Union[str, Path]] = None,
    max_workers: int = 4,
    chunk_size: int = 1000000,
) -> Any:
    """
    Add document content to agent prompts with improved reliability and performance.

    Args:
        agents: Dictionary mapping agent names to Agent objects
        docs: List of document paths
        doc_folder: Path to folder containing documents
        max_workers: Maximum number of parallel document processing workers
        chunk_size: Maximum characters to process at once to avoid memory issues

    Raises:
        ValueError: If neither docs nor doc_folder is provided
        RuntimeError: If document processing fails
    """
    if not agents:
        logger.warning(
            "No agents provided, skipping document distribution"
        )
        return

    if not docs and not doc_folder:
        logger.warning(
            "No documents or folder provided, skipping document distribution"
        )
        return

    logger.info("Starting document distribution to agents")

    try:
        processed_docs = []

        # Process individual documents in parallel
        if docs:
            with ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_doc = {
                    executor.submit(_process_document, doc): doc
                    for doc in docs
                }

                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        processed_docs.append(future.result())
                    except Exception as e:
                        logger.error(
                            f"Failed to process document {doc}: {str(e)}"
                        )
                        raise RuntimeError(
                            f"Document processing failed: {str(e)}"
                        )

        # Process folder if specified
        elif doc_folder:
            try:
                folder_content = doc_master(
                    folder_path=str(doc_folder), output_type="string"
                )
                processed_docs.append(folder_content)
            except Exception as e:
                logger.error(
                    f"Failed to process folder {doc_folder}: {str(e)}"
                )
                raise RuntimeError(
                    f"Folder processing failed: {str(e)}"
                )

        # Combine and chunk the processed documents
        combined_data = "\n".join(processed_docs)

        # Update agent prompts in chunks to avoid memory issues
        for agent in agents.values():
            try:
                for i in range(0, len(combined_data), chunk_size):
                    chunk = combined_data[i : i + chunk_size]
                    if i == 0:
                        agent.system_prompt += (
                            "\nDocuments:\n" + chunk
                        )
                    else:
                        agent.system_prompt += chunk
            except Exception as e:
                logger.error(
                    f"Failed to update agent prompt: {str(e)}"
                )
                raise RuntimeError(
                    f"Agent prompt update failed: {str(e)}"
                )

        logger.info(
            f"Successfully added documents to {len(agents)} agents"
        )

        return agents

    except Exception as e:
        logger.error(f"Document distribution failed: {str(e)}")
        raise RuntimeError(f"Document distribution failed: {str(e)}")
