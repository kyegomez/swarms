# MALT

*Specialized framework for complex language-based tasks and processing*

**Swarm Type**: `MALT`

## Overview

MALT (Multi-Agent Language Task) is a specialized framework optimized for complex language-based tasks, optimizing agent collaboration for sophisticated language processing operations. This architecture excels at tasks requiring deep linguistic analysis, natural language understanding, and complex text generation workflows.

Key features:
- **Language Optimization**: Specifically designed for natural language tasks
- **Linguistic Collaboration**: Agents work together on complex language operations
- **Text Processing Pipeline**: Structured approach to language task workflows
- **Advanced NLP**: Optimized for sophisticated language understanding tasks

## Use Cases

- Complex document analysis and processing
- Multi-language translation and localization
- Advanced content generation and editing
- Linguistic research and analysis tasks

## API Usage

### Basic MALT Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Legal Document Analysis MALT",
        "description": "Advanced linguistic analysis of legal documents using MALT framework",
        "swarm_type": "MALT",
        "task": "Perform comprehensive linguistic analysis of a complex legal contract including sentiment analysis, risk identification, clause categorization, and language complexity assessment",
        "agents": [
          {
            "agent_name": "Syntactic Analyzer",
            "description": "Analyzes sentence structure and grammar",
            "system_prompt": "You are a syntactic analysis expert. Analyze sentence structure, grammatical patterns, and linguistic complexity in legal texts.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          },
          {
            "agent_name": "Semantic Analyzer",
            "description": "Analyzes meaning and semantic relationships",
            "system_prompt": "You are a semantic analysis expert. Extract meaning, identify semantic relationships, and analyze conceptual content in legal documents.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Pragmatic Analyzer",
            "description": "Analyzes context and implied meanings",
            "system_prompt": "You are a pragmatic analysis expert. Analyze contextual meaning, implied obligations, and pragmatic implications in legal language.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.4
          },
          {
            "agent_name": "Discourse Analyzer",
            "description": "Analyzes document structure and flow",
            "system_prompt": "You are a discourse analysis expert. Analyze document structure, logical flow, and coherence in legal texts.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Risk Language Detector",
            "description": "Identifies risk-related language patterns",
            "system_prompt": "You are a legal risk language expert. Identify risk indicators, liability language, and potential legal concerns in contract language.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          }
        ],
        "max_loops": 1
      }'
    ```

=== "Python (requests)"
    ```python
    import requests
    import json

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    swarm_config = {
        "name": "Legal Document Analysis MALT",
        "description": "Advanced linguistic analysis of legal documents using MALT framework",
        "swarm_type": "MALT",
        "task": "Perform comprehensive linguistic analysis of a complex legal contract including sentiment analysis, risk identification, clause categorization, and language complexity assessment",
        "agents": [
            {
                "agent_name": "Syntactic Analyzer",
                "description": "Analyzes sentence structure and grammar",
                "system_prompt": "You are a syntactic analysis expert. Analyze sentence structure, grammatical patterns, and linguistic complexity in legal texts.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            },
            {
                "agent_name": "Semantic Analyzer",
                "description": "Analyzes meaning and semantic relationships",
                "system_prompt": "You are a semantic analysis expert. Extract meaning, identify semantic relationships, and analyze conceptual content in legal documents.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Pragmatic Analyzer",
                "description": "Analyzes context and implied meanings",
                "system_prompt": "You are a pragmatic analysis expert. Analyze contextual meaning, implied obligations, and pragmatic implications in legal language.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.4
            },
            {
                "agent_name": "Discourse Analyzer",
                "description": "Analyzes document structure and flow",
                "system_prompt": "You are a discourse analysis expert. Analyze document structure, logical flow, and coherence in legal texts.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Risk Language Detector",
                "description": "Identifies risk-related language patterns",
                "system_prompt": "You are a legal risk language expert. Identify risk indicators, liability language, and potential legal concerns in contract language.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            }
        ],
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("MALT framework completed successfully!")
        print(f"Linguistic analysis: {result['output']['linguistic_analysis']}")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "legal-document-analysis-malt",
  "swarm_type": "MALT",
  "task": "Perform comprehensive linguistic analysis of a complex legal contract including sentiment analysis, risk identification, clause categorization, and language complexity assessment",
  "output": {
    "linguistic_analysis": {
      "syntactic_analysis": {
        "complexity_score": 8.2,
        "sentence_structure": "Predominantly complex and compound-complex sentences",
        "grammatical_patterns": "Heavy use of passive voice, subordinate clauses, and technical terminology",
        "readability": "Graduate level (16+ years of education required)"
      },
      "semantic_analysis": {
        "key_concepts": ["liability", "indemnification", "force majeure", "intellectual property"],
        "semantic_relationships": "Strong hierarchical concept relationships with clear definitional structures",
        "conceptual_density": "High - 3.2 legal concepts per sentence average",
        "ambiguity_indicators": ["potentially", "reasonable efforts", "material adverse effect"]
      },
      "pragmatic_analysis": {
        "implied_obligations": [
          "Good faith performance expected",
          "Timely notice requirements implied",
          "Mutual cooperation assumed"
        ],
        "power_dynamics": "Balanced with slight advantage to service provider",
        "speech_acts": "Predominantly commissives (commitments) and directives (obligations)"
      },
      "discourse_analysis": {
        "document_structure": "Well-organized with clear section hierarchy",
        "logical_flow": "Sequential with appropriate cross-references",
        "coherence_score": 8.5,
        "transition_patterns": "Formal legal transitions with clause numbering"
      },
      "risk_language": {
        "high_risk_terms": ["unlimited liability", "personal guarantee", "joint and several"],
        "risk_mitigation_language": ["subject to", "limited to", "except as provided"],
        "liability_indicators": 23,
        "risk_level": "Medium-High"
      }
    },
    "comprehensive_summary": {
      "language_complexity": "High complexity legal document requiring specialized knowledge",
      "risk_assessment": "Medium-high risk with standard legal protections",
      "readability_concerns": "Requires legal expertise for full comprehension",
      "recommendations": [
        "Consider plain language summary for key terms",
        "Review unlimited liability clauses",
        "Clarify ambiguous terms identified"
      ]
    }
  },
  "metadata": {
    "malt_framework": {
      "linguistic_layers_analyzed": 5,
      "language_processing_depth": "Advanced multi-layer analysis",
      "specialized_nlp_operations": [
        "Syntactic parsing",
        "Semantic role labeling", 
        "Pragmatic inference",
        "Discourse segmentation",
        "Risk pattern recognition"
      ]
    },
    "execution_time_seconds": 35.7,
    "billing_info": {
      "total_cost": 0.089
    }
  }
}
```

## Best Practices

- Use MALT for sophisticated language processing tasks
- Design agents with complementary linguistic analysis capabilities
- Ideal for tasks requiring deep language understanding
- Consider multiple levels of linguistic analysis (syntax, semantics, pragmatics)

## Related Swarm Types

- [SequentialWorkflow](sequential_workflow.md) - For ordered language processing
- [MixtureOfAgents](mixture_of_agents.md) - For diverse linguistic expertise
- [HierarchicalSwarm](hierarchical_swarm.md) - For structured language analysis