
from swarms.structs.agent import Agent
from swarms.utils.pdf_to_text import pdf_to_text
import asyncio

class DocumentProcessingPipeline:
    def __init__(self):
        self.document_analyzer = Agent(
            agent_name="Document-Analyzer",
            agent_description="Enterprise document analysis specialist",
            system_prompt="""You are an expert document analyzer specializing in:
            1. Complex Document Structure Analysis
            2. Key Information Extraction
            3. Compliance Verification
            4. Document Classification
            5. Content Validation""",
            max_loops=2,
            model_name="gpt-4"
        )
        
        self.legal_reviewer = Agent(
            agent_name="Legal-Reviewer",
            agent_description="Legal compliance and risk assessment specialist",
            system_prompt="""You are a legal review expert focusing on:
            1. Regulatory Compliance Check
            2. Legal Risk Assessment
            3. Contractual Obligation Analysis
            4. Privacy Requirement Verification
            5. Legal Term Extraction""",
            max_loops=2,
            model_name="gpt-4"
        )
        
        self.data_extractor = Agent(
            agent_name="Data-Extractor",
            agent_description="Structured data extraction specialist",
            system_prompt="""You are a data extraction expert specializing in:
            1. Named Entity Recognition
            2. Relationship Extraction
            3. Tabular Data Processing
            4. Metadata Extraction
            5. Data Standardization""",
            max_loops=2,
            model_name="gpt-4"
        )

    async def process_document(self, document_path):
        # Convert document to text
        document_text = pdf_to_text(document_path)
        
        # Parallel processing tasks
        tasks = [
            self.document_analyzer.arun(f"Analyze this document: {document_text}"),
            self.legal_reviewer.arun(f"Review legal aspects: {document_text}"),
            self.data_extractor.arun(f"Extract structured data: {document_text}")
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            "document_analysis": results[0],
            "legal_review": results[1],
            "extracted_data": results[2]
        }

# Usage
processor = DocumentProcessingPipeline()
