import concurrent.futures
import os
from typing import Union, List
import PyPDF2
import markdown
from pathlib import Path
from swarms.utils.litellm_tokenizer import count_tokens
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.formatter import formatter


class LongAgent:
    """
    A class to handle and process long-form content from various sources including PDFs,
    markdown files, and large text documents.
    """

    def __init__(
        self,
        name: str = "LongAgent",
        description: str = "A long-form content processing agent",
        token_count_per_agent: int = 16000,
        output_type: str = "final",
        model_name: str = "gpt-4o-mini",
        aggregator_model_name: str = "gpt-4o-mini",
    ):
        """Initialize the LongAgent."""
        self.name = name
        self.description = description
        self.model_name = model_name
        self.aggregator_model_name = aggregator_model_name
        self.content = ""
        self.metadata = {}
        self.token_count_per_agent = token_count_per_agent
        self.output_type = output_type
        self.agents = []
        self.conversation = Conversation()

    def load_pdf(self, file_path: Union[str, Path]) -> str:
        """
        Load and extract text from a PDF file.

        Args:
            file_path (Union[str, Path]): Path to the PDF file

        Returns:
            str: Extracted text from the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"PDF file not found at {file_path}"
            )

        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()

        self.content = text
        self.metadata["source"] = "pdf"
        self.metadata["file_path"] = str(file_path)
        return text

    def load_markdown(self, file_path: Union[str, Path]) -> str:
        """
        Load and process a markdown file.

        Args:
            file_path (Union[str, Path]): Path to the markdown file

        Returns:
            str: Processed markdown content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Markdown file not found at {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Convert markdown to HTML for processing
        markdown.markdown(content)

        self.content = content
        self.metadata["source"] = "markdown"
        self.metadata["file_path"] = str(file_path)
        return content

    def load_text(self, text: str) -> str:
        """
        Load and process a large text string.

        Args:
            text (str): The text content to process

        Returns:
            str: The processed text
        """
        self.content = text
        self.metadata["source"] = "text"
        return text

    def get_content(self) -> str:
        """
        Get the current content being processed.

        Returns:
            str: The current content
        """
        return self.content

    def get_metadata(self) -> dict:
        """
        Get the metadata associated with the current content.

        Returns:
            dict: The metadata dictionary
        """
        return self.metadata

    def count_token_document(
        self, file_path: Union[str, Path]
    ) -> int:
        """
        Count the number of tokens in a document.

        Args:
            document (str): The document to count tokens for
        """
        if file_path.endswith(".pdf"):
            count = count_tokens(self.load_pdf(file_path))
            formatter.print_panel(
                f"Token count for {file_path}: {count}",
                title="Token Count",
            )
            print(f"Token count for {file_path}: {count}")
        elif file_path.endswith(".md"):
            count = count_tokens(self.load_markdown(file_path))
            formatter.print_panel(
                f"Token count for {file_path}: {count}",
                title="Token Count",
            )
            print(f"Token count for {file_path}: {count}")
        elif file_path.endswith(".txt"):
            count = count_tokens(self.load_text(file_path))
            formatter.print_panel(
                f"Token count for {file_path}: {count}",
                title="Token Count",
            )
            print(f"Token count for {file_path}: {count}")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return count

    def count_multiple_documents(
        self, file_paths: List[Union[str, Path]]
    ) -> int:
        """
        Count the number of tokens in multiple documents.

        Args:
            file_paths (List[Union[str, Path]]): The list of file paths to count tokens for

        Returns:
            int: Total token count across all documents
        """
        total_tokens = 0
        # Calculate max_workers as 20% of CPU count
        max_workers = max(1, int(os.cpu_count() * 0.2))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(self.count_token_document, file_path)
                for file_path in file_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    total_tokens += future.result()
                except Exception as e:
                    formatter.print_panel(
                        f"Error processing document: {str(e)}",
                        title="Error",
                    )
                    continue
        return total_tokens

    def create_agents_for_documents(
        self, file_paths: List[Union[str, Path]]
    ) -> List[Agent]:
        """
        Create agents for each document chunk and process them.

        Args:
            file_paths (List[Union[str, Path]]): The list of file paths to create agents for

        Returns:
            List[Agent]: List of created agents
        """
        for file_path in file_paths:
            # Load the document content
            if str(file_path).endswith(".pdf"):
                content = self.load_pdf(file_path)
            elif str(file_path).endswith(".md"):
                content = self.load_markdown(file_path)
            else:
                content = self.load_text(str(file_path))

            # Split content into chunks based on token count
            chunks = self._split_into_chunks(content)

            # Create an agent for each chunk
            for i, chunk in enumerate(chunks):
                agent = Agent(
                    agent_name=f"Document Analysis Agent - {Path(file_path).name} - Chunk {i+1}",
                    system_prompt="""
                    You are an expert document analysis and summarization agent specialized in processing and understanding complex documents. Your primary responsibilities include:

                    1. Document Analysis:
                    - Thoroughly analyze the provided document chunk
                    - Identify key themes, main arguments, and important details
                    - Extract critical information and relationships between concepts

                    2. Summarization Capabilities:
                    - Create concise yet comprehensive summaries
                    - Generate both high-level overviews and detailed breakdowns
                    - Highlight key points, findings, and conclusions
                    - Maintain context and relationships between different sections

                    3. Information Extraction:
                    - Identify and extract important facts, figures, and data points
                    - Recognize and preserve technical terminology and domain-specific concepts
                    - Maintain accuracy in representing the original content

                    4. Response Format:
                    - Provide clear, structured responses
                    - Use bullet points for key findings
                    - Include relevant quotes or references when necessary
                    - Maintain professional and academic tone

                    5. Context Awareness:
                    - Consider the document's purpose and target audience
                    - Adapt your analysis based on the document type (academic, technical, general)
                    - Preserve the original meaning and intent

                    Your goal is to help users understand and extract value from this document chunk while maintaining accuracy and completeness in your analysis.
                    """,
                    model_name=self.model_name,
                    max_loops=1,
                    max_tokens=self.token_count_per_agent,
                )

                # Run the agent on the chunk
                output = agent.run(
                    f"Please analyze and summarize the following document chunk:\n\n{chunk}"
                )

                # Add the output to the conversation
                self.conversation.add(
                    role=agent.agent_name,
                    content=output,
                )

                self.agents.append(agent)

        return self.agents

    def _split_into_chunks(self, content: str) -> List[str]:
        """
        Split content into chunks based on token count.

        Args:
            content (str): The content to split

        Returns:
            List[str]: List of content chunks
        """
        chunks = []
        current_chunk = ""
        current_tokens = 0

        # Split content into sentences (simple approach)
        sentences = content.split(". ")

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            if (
                current_tokens + sentence_tokens
                > self.token_count_per_agent
            ):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += (
                    ". " + sentence if current_chunk else sentence
                )
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def count_total_agents(self) -> int:
        """
        Count the total number of agents.
        """
        count = len(self.agents)
        formatter.print_panel(f"Total agents created: {count}")
        return count

    def _create_aggregator_agent(self) -> Agent:
        """
        Create an aggregator agent for synthesizing document summaries.

        Returns:
            Agent: The configured aggregator agent
        """
        return Agent(
            agent_name="Document Aggregator Agent",
            system_prompt="""
            You are an expert document synthesis agent specialized in creating comprehensive reports from multiple document summaries. Your responsibilities include:

            1. Synthesis and Integration:
            - Combine multiple document summaries into a coherent narrative
            - Identify and resolve any contradictions or inconsistencies
            - Maintain logical flow and structure in the final report
            - Preserve important details while eliminating redundancy

            2. Report Structure:
            - Create a clear, hierarchical structure for the report
            - Include an executive summary at the beginning
            - Organize content into logical sections with clear headings
            - Ensure smooth transitions between different topics

            3. Analysis and Insights:
            - Identify overarching themes and patterns across summaries
            - Draw meaningful conclusions from the combined information
            - Highlight key findings and their implications
            - Provide context and connections between different pieces of information

            4. Quality Assurance:
            - Ensure factual accuracy and consistency
            - Maintain professional and academic tone
            - Verify that all important information is included
            - Check for clarity and readability

            Your goal is to create a comprehensive, well-structured report that effectively synthesizes all the provided document summaries into a single coherent document.
            """,
            model_name=self.aggregator_model_name,
            max_loops=1,
            max_tokens=self.token_count_per_agent,
        )

    def run(self, file_paths: List[Union[str, Path]]) -> str:
        """
        Run the document processing pipeline and generate a comprehensive report.

        Args:
            file_paths (List[Union[str, Path]]): The list of file paths to process

        Returns:
            str: The final comprehensive report
        """
        # Count total tokens
        total_tokens = self.count_multiple_documents(file_paths)
        formatter.print_panel(
            f"Total tokens: {total_tokens}", title="Total Tokens"
        )

        total_amount_of_agents = (
            total_tokens / self.token_count_per_agent
        )
        formatter.print_panel(
            f"Total amount of agents: {total_amount_of_agents}",
            title="Total Amount of Agents",
        )

        # First, process all documents and create chunk agents
        self.create_agents_for_documents(file_paths)

        # Format the number of agents
        # formatter.print_panel(f"Number of agents: {len(self.agents)}", title="Number of Agents")

        # Create aggregator agent and collect summaries
        aggregator_agent = self._create_aggregator_agent()
        combined_summaries = self.conversation.get_str()

        # Generate the final comprehensive report
        final_report = aggregator_agent.run(
            f"""
            Please create a comprehensive report by synthesizing the following document summaries:

            {combined_summaries}

            Please structure your response as follows:
            1. Executive Summary
            2. Main Findings and Analysis
            3. Key Themes and Patterns
            4. Detailed Breakdown by Topic
            5. Conclusions and Implications

            Ensure the report is well-organized, comprehensive, and maintains a professional tone throughout.
            """
        )

        # Add the final report to the conversation
        self.conversation.add(
            role="Document Aggregator Agent", content=final_report
        )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )
