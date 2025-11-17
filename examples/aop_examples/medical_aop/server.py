# Import medical agents defined in the demo module
from examples.demos.medical.medical_coder_agent import (
    chief_medical_officer,
    internist,
    medical_coder,
    synthesizer,
    virologist,
)
from swarms.structs.aop import AOP


def _enrich_agents_metadata() -> None:
    """
    Add lightweight tags/capabilities/roles to imported agents for
    better discovery results.
    """
    chief_medical_officer.tags = [
        "coordination",
        "diagnosis",
        "triage",
    ]
    chief_medical_officer.capabilities = [
        "case-intake",
        "differential",
        "planning",
    ]
    chief_medical_officer.role = "coordinator"

    virologist.tags = ["virology", "infectious-disease"]
    virologist.capabilities = ["viral-analysis", "icd10-suggestion"]
    virologist.role = "specialist"

    internist.tags = ["internal-medicine", "evaluation"]
    internist.capabilities = [
        "system-review",
        "hcc-tagging",
        "risk-stratification",
    ]
    internist.role = "specialist"

    medical_coder.tags = ["coding", "icd10", "compliance"]
    medical_coder.capabilities = [
        "code-assignment",
        "documentation-review",
    ]
    medical_coder.role = "coder"

    synthesizer.tags = ["synthesis", "reporting"]
    synthesizer.capabilities = [
        "evidence-reconciliation",
        "final-report",
    ]
    synthesizer.role = "synthesizer"


def _medical_input_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Patient case or instruction for the agent",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Processing priority",
            },
            "include_images": {
                "type": "boolean",
                "description": "Whether to consider linked images if provided",
                "default": False,
            },
            "img": {
                "type": "string",
                "description": "Optional image path/URL",
            },
            "imgs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of images",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }


def _medical_output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "success": {"type": "boolean"},
            "error": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Optional confidence in the assessment",
            },
            "codes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of suggested ICD-10 codes",
            },
        },
        "required": ["result", "success"],
        "additionalProperties": True,
    }


def main() -> None:
    """
    Start an AOP MCP server that exposes the medical agents as tools with
    structured schemas and per-agent settings.
    """
    _enrich_agents_metadata()

    deployer = AOP(
        server_name="Medical-AOP-Server",
        port=8000,
        verbose=False,
        traceback_enabled=True,
        log_level="INFO",
        transport="streamable-http",
    )

    input_schema = _medical_input_schema()
    output_schema = _medical_output_schema()

    # Register each agent with a modest, role-appropriate timeout
    deployer.add_agent(
        chief_medical_officer,
        timeout=45,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    deployer.add_agent(
        virologist,
        timeout=40,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    deployer.add_agent(
        internist,
        timeout=40,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    deployer.add_agent(
        medical_coder,
        timeout=50,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    deployer.add_agent(
        synthesizer,
        timeout=45,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    deployer.run()


if __name__ == "__main__":
    main()
