import concurrent.futures
import json
import os
import time
import uuid
from io import BytesIO
from typing import Dict, List, Union

import PyPDF2
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from swarms import Agent


def user_id_generator():
    return str(uuid.uuid4().hex)


timestamp = time.strftime("%Y%m%d_%H%M%S")
# print(timestamp)


class MortgageApplicationInput(BaseModel):
    user_id: str = Field(default_factory=user_id_generator)
    timestamp: str = Field(default_factory=timestamp)
    application_data: str = Field(
        description="The raw text of the mortgage application."
    )


class MortgageApplicationOutput(BaseModel):
    user_id: str = Field(default_factory=user_id_generator)
    input_data: MortgageApplicationInput = Field(
        description="The input data for the mortgage application."
    )
    document_analysis: str = Field(
        description="The structured analysis of the mortgage application."
    )
    risk_evaluation: str = Field(
        description="The risk evaluation of the mortgage application."
    )
    underwriting_decision: str = Field(
        description="The underwriting decision of the mortgage application."
    )


def clean_markdown(text: str) -> str:
    """
    Removes all markdown symbols from text.

    Args:
        text (str): Text containing markdown symbols

    Returns:
        str: Text with markdown symbols removed
    """
    markdown_symbols = [
        "```markdown",
        "```",
        "#",
        "*",
        "_",
        "`",
        ">",
        "-",
        "+",
        "[",
        "]",
        "(",
        ")",
        "|",
    ]
    cleaned_text = text
    for symbol in markdown_symbols:
        cleaned_text = cleaned_text.replace(symbol, "")
    return cleaned_text.strip()


class MortgageUnderwritingSwarm:
    def __init__(
        self,
        user_id: str = user_id_generator(),
        save_directory: str = "./autosave",
        return_format: str = "pdf",
    ):
        """
        Initialize the MortgageUnderwritingSwarm with the necessary Agents.
        Args:
            save_directory (str): Directory where intermediate results and final documents will be autosaved.
        """
        self.user_id = user_id
        self.save_directory = save_directory
        self.return_format = return_format
        os.makedirs(self.save_directory, exist_ok=True)

        # -------------------------------
        # 1) Document Analyzer Agent
        # -------------------------------
        self.document_agent = Agent(
            agent_name="Document-Analyzer-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        )
        self.document_prompt = """
        You are a highly experienced Mortgage Document Analysis Expert with deep knowledge of federal and state mortgage regulations. Your task is to:

        1. Parse and extract key data from unstructured documents (PDF or text) while ensuring compliance with:
        - Truth in Lending Act (TILA) requirements
        - Real Estate Settlement Procedures Act (RESPA) guidelines
        - Fair Credit Reporting Act (FCRA) standards
        - Equal Credit Opportunity Act (ECOA) requirements

        2. Validate data consistency and regulatory compliance for:
        - Income verification (including all sources of income)
        - Credit scores and credit history
        - Property details and appraisal information
        - Debt obligations and payment history
        - Employment verification
        - Asset documentation
        - Identity verification documents

        3. Highlight any discrepancies, red flags, or potential compliance violations, including:
        - Inconsistencies in reported income vs documentation
        - Suspicious patterns in bank statements
        - Potential identity theft indicators
        - Missing required regulatory disclosures
        - Fair lending concerns
        - Anti-money laundering (AML) red flags

        4. Provide a comprehensive, well-structured summary that includes:
        - All key findings organized by category
        - Compliance checklist results
        - Documentation completeness assessment
        - Regulatory disclosure verification
        - Quality control notes

        5. Clearly indicate any missing or ambiguous information required by:
        - Federal regulations
        - State-specific requirements
        - Agency guidelines (FHA, VA, Fannie Mae, Freddie Mac)
        - Internal compliance policies

        6. Format output in a standardized structure that:
        - Facilitates automated compliance checks
        - Enables clear audit trails
        - Supports regulatory reporting requirements
        - Can be easily consumed by subsequent agents
        """

        # -------------------------------
        # 2) Risk Evaluator Agent
        # -------------------------------
        self.risk_agent = Agent(
            agent_name="Risk-Evaluator-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        )
        self.risk_prompt = """
        You are an expert Risk Evaluator for mortgage applications with comprehensive knowledge of regulatory compliance. Your responsibilities:

        1. Conduct thorough risk assessment in accordance with:
        - Dodd-Frank Act requirements
        - Consumer Financial Protection Bureau (CFPB) guidelines
        - Federal Reserve Board regulations
        - Agency-specific requirements (FHA, VA, Fannie Mae, Freddie Mac)

        2. Evaluate key risk factors including:
        - Debt-to-income ratio (DTI) compliance with QM rules
        - Credit history analysis per FCRA guidelines
        - Property valuation in line with USPAP standards
        - Income stability and verification per agency requirements
        - Assets and reserves adequacy
        - Employment history and verification
        - Occupancy risk assessment
        - Property type and use restrictions

        3. Calculate and assign risk scores:
        - Overall application risk score (1-10 scale)
        - Individual component risk scores
        - Regulatory compliance risk assessment
        - Fraud risk indicators
        - Default risk probability

        4. Identify and document:
        - High-risk elements requiring additional scrutiny
        - Potential regulatory compliance issues
        - Required compensating factors
        - Secondary market eligibility concerns
        - Fair lending considerations

        5. Recommend risk mitigation strategies:
        - Additional documentation requirements
        - Income/asset verification needs
        - Compensating factor documentation
        - Alternative qualification approaches
        - Regulatory compliance remediation steps

        6. Generate comprehensive risk analysis including:
        - Detailed risk assessment findings
        - Compliance verification results
        - Supporting documentation requirements
        - Clear justification for all conclusions
        - Regulatory requirement adherence confirmation
        """

        # -------------------------------
        # 3) Mortgage Underwriter Agent
        # -------------------------------
        self.underwriter_agent = Agent(
            agent_name="Mortgage-Underwriter-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        )
        self.underwriter_prompt = """
        You are a seasoned Mortgage Underwriter with expertise in regulatory compliance and industry standards. Your role is to:

        1. Make final underwriting decisions while ensuring compliance with:
        - Qualified Mortgage (QM) and Ability-to-Repay (ATR) rules
        - Fair lending laws (ECOA, FHA, HMDA)
        - Agency guidelines (FHA, VA, Fannie Mae, Freddie Mac)
        - State-specific lending requirements
        - Internal credit policies and procedures

        2. Review and synthesize:
        - Document Analyzer findings
        - Risk Evaluator assessments
        - Compliance verification results
        - Quality control checks
        - Regulatory requirements
        - Secondary market guidelines

        3. Determine appropriate decision category:
        - Approved
        - Conditionally Approved (with specific conditions)
        - Denied (with detailed adverse action notice requirements)
        - Counteroffer recommendations
        - Alternative program suggestions

        4. For all decisions, provide:
        - Clear written justification
        - Regulatory compliance confirmation
        - Required disclosures identification
        - Adverse action notices if required
        - Fair lending analysis documentation
        - Secondary market eligibility determination

        5. For conditional approvals, specify:
        - Required documentation
        - Timeline requirements
        - Regulatory compliance conditions
        - Prior-to-funding conditions
        - Post-closing requirements
        - Quality control conditions

        6. Generate comprehensive decision report including:
        - Detailed underwriting analysis
        - Compliance verification results
        - Supporting documentation list
        - Condition status tracking
        - Regulatory requirement satisfaction
        - Clear audit trail documentation

        7. Ensure all decisions adhere to:
        - Fair lending requirements
        - Anti-discrimination laws
        - UDAAP regulations
        - State and federal disclosure requirements
        - Agency and investor guidelines
        - Internal policies and procedures
        """

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------
    def pdf_to_text(self, pdf_file_path: str) -> str:
        """
        Converts a PDF file to a string by extracting its text content.
        Args:
            pdf_file_path (str): The path to the PDF file.
        Returns:
            str: The extracted text from the PDF.
        """
        text_content = []
        with open(pdf_file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_content.append(page_text)
        return "\n".join(text_content)

    def autosave_result(
        self, result_data: str, filename: str
    ) -> None:
        """
        Autosaves intermediate or final results to a text file in the designated directory.
        Args:
            result_data (str): The data to be written to the file.
            filename (str): The desired filename (without path).
        """
        full_path = os.path.join(self.save_directory, filename)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(result_data)

    def generate_pdf_report(
        self, content: str, pdf_path: str
    ) -> None:
        """
        Generates a simple PDF report from text content using ReportLab.
        Args:
            content (str): The textual content for the PDF.
            pdf_path (str): Where to save the generated PDF.
        """
        BytesIO()
        c = canvas.Canvas(pdf_path, pagesize=LETTER)
        width, height = LETTER

        # Simple text wrap by splitting lines
        lines = clean_markdown(content).split("\n")
        current_height = height - 50  # top margin

        for line in lines:
            # If the line is too long, wrap it manually (simple approach)
            max_chars = 90  # approx number of characters per line for LETTER size
            while len(line) > max_chars:
                c.drawString(50, current_height, line[:max_chars])
                line = line[max_chars:]
                current_height -= 15  # line spacing
            c.drawString(50, current_height, line)
            current_height -= 15

            # Add a new page if we go beyond the margin
            if current_height <= 50:
                c.showPage()
                current_height = height - 50

        c.save()

    # --------------------------------------------------------------------------
    # Core Processing Methods
    # --------------------------------------------------------------------------
    def analyze_documents(self, document_data: str) -> str:
        """
        Runs the Document Analyzer Agent on the given data.
        Args:
            document_data (str): Text representing the mortgage documents.
        Returns:
            str: Structured summary and highlights from the document analysis.
        """
        prompt_input = (
            self.document_prompt
            + "\n\n--- BEGIN DOCUMENTS ---\n"
            + document_data
            + "\n--- END DOCUMENTS ---\n"
        )
        print("Running Document Analyzer Agent...")
        result = self.document_agent.run(prompt_input)
        self.autosave_result(result, "document_analysis.txt")
        return result

    def evaluate_risk(self, document_analysis: str) -> str:
        """
        Runs the Risk Evaluator Agent using the results from the Document Analyzer.
        Args:
            document_analysis (str): The structured analysis from the Document Analyzer.
        Returns:
            str: Risk analysis including risk score and explanation.
        """
        prompt_input = (
            self.risk_prompt
            + "\n\n--- DOCUMENT ANALYSIS OUTPUT ---\n"
            + document_analysis
            + "\n--- END ANALYSIS OUTPUT ---\n"
        )
        print("Running Risk Evaluator Agent...")
        result = self.risk_agent.run(prompt_input)
        self.autosave_result(result, "risk_evaluation.txt")
        return result

    def underwrite_mortgage(
        self, document_analysis: str, risk_evaluation: str
    ) -> str:
        """
        Runs the Mortgage Underwriter Agent to produce the final underwriting decision.
        Args:
            document_analysis (str): Output from the Document Analyzer.
            risk_evaluation (str): Output from the Risk Evaluator.
        Returns:
            str: Final decision text with rationale.
        """
        prompt_input = (
            self.underwriter_prompt
            + "\n\n--- DOCUMENT ANALYSIS SUMMARY ---\n"
            + document_analysis
            + "\n--- RISK EVALUATION REPORT ---\n"
            + risk_evaluation
            + "\n--- END REPORTS ---\n"
        )
        print("Running Mortgage Underwriter Agent...")
        result = self.underwriter_agent.run(prompt_input)
        self.autosave_result(result, "underwriting_decision.txt")
        return result

    # --------------------------------------------------------------------------
    # High-Level Workflow
    # --------------------------------------------------------------------------
    def run(
        self,
        application_data: str,
        return_format: str = "pdf",
        output_filename: str = "UnderwritingDecision",
    ) -> Union[str, Dict]:
        """
        Processes a single mortgage application from documents to final underwriting decision.
        Allows returning data in either PDF or JSON format.

        Args:
            application_data (str): The text representation of the applicant’s documents.
            return_format (str): "pdf" or "json". Defaults to "pdf".
            output_filename (str): Base filename (without extension) for the output file.

        Returns:
            Union[str, Dict]: If return_format="json", returns a dict with the final data.
                              If return_format="pdf", returns the path of the generated PDF.
        """
        # Step 1: Document Analysis
        doc_analysis = self.analyze_documents(application_data)

        # Step 2: Risk Evaluation
        risk_eval = self.evaluate_risk(doc_analysis)

        # Step 3: Underwriting Decision
        final_decision = self.underwrite_mortgage(
            doc_analysis, risk_eval
        )

        # Prepare final content (text)
        final_content = (
            "---- Mortgage Underwriting Decision Report ----\n\n"
            "DOCUMENT ANALYSIS:\n" + doc_analysis + "\n\n"
            "RISK EVALUATION:\n" + risk_eval + "\n\n"
            "FINAL UNDERWRITING DECISION:\n" + final_decision + "\n"
        )

        # Return JSON
        if return_format.lower() == "json":
            output_data = {
                "document_analysis": doc_analysis,
                "risk_evaluation": risk_eval,
                "final_decision": final_decision,
            }
            json_path = os.path.join(
                self.save_directory, f"{output_filename}.json"
            )
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(output_data, jf, indent=2)
            return output_data

        # Generate PDF
        elif return_format.lower() == "pdf":
            pdf_path = os.path.join(
                self.save_directory, f"{output_filename}.pdf"
            )
            self.generate_pdf_report(final_content, pdf_path)
            return pdf_path

        else:
            raise ValueError(
                "Invalid return format. Choose either 'pdf' or 'json'."
            )

    def run_concurrently(
        self,
        application_data: str,
        return_format: str = "pdf",
        output_filename: str = "UnderwritingDecision",
    ) -> Union[str, Dict]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(
                    self.run,
                    application_data,
                    return_format,
                    output_filename,
                )
            ]
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
        return results

    # --------------------------------------------------------------------------
    # Batch Processing
    # --------------------------------------------------------------------------
    def runs_in_batch(
        self,
        list_of_application_data: List[str],
        return_format: str = "pdf",
    ) -> List[Union[str, Dict]]:
        """
        Processes multiple mortgage applications in a batch and returns the results as
        either PDFs or JSON structures for each application.

        Args:
            list_of_application_data (List[str]): A list of string representations
                                                  of mortgage applications (e.g., raw text).
            return_format (str): "pdf" or "json" format for the output files.

        Returns:
            List[Union[str, Dict]]: A list of outputs (either file paths to PDFs or JSON dicts).
        """
        results = []
        for idx, application_text in enumerate(
            list_of_application_data, start=1
        ):
            output_filename = f"UnderwritingDecision_{idx}"
            print(f"\n--- Processing Application {idx} ---")
            result = self.run(
                application_data=application_text,
                return_format=return_format,
                output_filename=output_filename,
            )
            results.append(result)
        return results

    # --------------------------------------------------------------------------
    # PDF/Document Conversion Helpers
    # --------------------------------------------------------------------------
    def convert_pdfs_to_texts(
        self, pdf_paths: List[str]
    ) -> List[str]:
        """
        Converts multiple PDFs into text.

        Args:
            pdf_paths (List[str]): A list of file paths to PDF documents.

        Returns:
            List[str]: A list of extracted text contents, one per PDF in the list.
        """
        text_results = []
        for pdf_path in pdf_paths:
            print(f"Converting PDF to text: {pdf_path}")
            text_data = self.pdf_to_text(pdf_path)
            text_results.append(text_data)
        return text_results


# ------------------------------------------------------------------------------
# Example Usage (As a Script)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Sample mortgage application text (or read from PDF, DB, etc.)
    sample_application_data = """
    Mortgage Application Data:
    Applicant Name: Jane Doe
    DOB: 02/14/1985
    SSN: 987-65-4321
    Annual Income: $95,000
    Credit Score: 690
    Outstanding Debt: $40,000
    Property Appraisal: $300,000
    Loan Amount Request: $270,000
    Employment: 3+ years at current employer
    Bank Statements & Tax Returns: Provided for the last year
    Extra Notes: Some minor late payments on credit cards in 2020.
    """

    # Initialize the swarm
    swarm = MortgageUnderwritingSwarm(
        save_directory="./autosave_results"
    )

    # 1) Convert PDF to text if needed
    # pdf_text = swarm.pdf_to_text("path_to_some_pdf.pdf")
    # Or convert multiple PDFs in batch
    # texts_from_pdfs = swarm.convert_pdfs_to_texts(["file1.pdf", "file2.pdf"])

    # 2) Process a single application
    final_pdf_path = swarm.run(
        application_data=sample_application_data,
        return_format="pdf",  # or "json"
        output_filename="JaneDoe_UnderwritingDecision",
    )
    print(f"PDF generated at: {final_pdf_path}")

    # 3) Process multiple applications in a batch
    # multiple_apps = [sample_application_data, sample_application_data]  # Pretend we have 2
    # batch_results = swarm.runs_in_batch(
    #     multiple_apps,
    #     return_format="json"
    # )
    # Each item in batch_results will be a JSON dict if return_format="json".
    # print("\nBatch Processing Results (JSON):")
    # for result in batch_results:
    #     print(json.dumps(result, indent=2))
