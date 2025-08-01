# SpreadSheetSwarm

*Structured approach to data management and operations in spreadsheet-like format*

**Swarm Type**: `SpreadSheetSwarm`

## Overview

The SpreadSheetSwarm provides a structured approach to data management and operations, ideal for tasks involving data analysis, transformation, and systematic processing in a spreadsheet-like structure. This architecture organizes agents to work on data in a tabular format with clear rows, columns, and processing workflows.

Key features:
- **Structured Data Processing**: Organizes work in spreadsheet-like rows and columns
- **Systematic Operations**: Sequential and methodical data handling
- **Data Transformation**: Efficient processing of structured datasets
- **Collaborative Analysis**: Multiple agents working on different data aspects

## Use Cases

- Financial data analysis and reporting
- Customer data processing and segmentation
- Inventory management and tracking
- Research data compilation and analysis

## API Usage

### Basic SpreadSheetSwarm Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Financial Analysis Spreadsheet",
        "description": "Systematic financial data analysis using spreadsheet structure",
        "swarm_type": "SpreadSheetSwarm",
        "task": "Analyze quarterly financial performance data for a retail company with multiple product lines and create comprehensive insights",
        "agents": [
          {
            "agent_name": "Data Validator",
            "description": "Validates and cleans financial data",
            "system_prompt": "You are a data validation specialist. Clean, validate, and structure financial data ensuring accuracy and consistency.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          },
          {
            "agent_name": "Revenue Analyst",
            "description": "Analyzes revenue trends and patterns",
            "system_prompt": "You are a revenue analyst. Focus on revenue trends, growth patterns, and seasonal variations across product lines.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Cost Analyst",
            "description": "Analyzes cost structures and margins",
            "system_prompt": "You are a cost analyst. Examine cost structures, margin analysis, and expense categorization.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Performance Calculator",
            "description": "Calculates KPIs and financial metrics",
            "system_prompt": "You are a financial metrics specialist. Calculate KPIs, ratios, and performance indicators from the analyzed data.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.1
          },
          {
            "agent_name": "Report Generator",
            "description": "Creates structured financial reports",
            "system_prompt": "You are a report generator. Create comprehensive, well-structured financial reports with insights and recommendations.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.4
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
        "name": "Financial Analysis Spreadsheet",
        "description": "Systematic financial data analysis using spreadsheet structure",
        "swarm_type": "SpreadSheetSwarm",
        "task": "Analyze quarterly financial performance data for a retail company with multiple product lines and create comprehensive insights",
        "agents": [
            {
                "agent_name": "Data Validator",
                "description": "Validates and cleans financial data",
                "system_prompt": "You are a data validation specialist. Clean, validate, and structure financial data ensuring accuracy and consistency.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            },
            {
                "agent_name": "Revenue Analyst",
                "description": "Analyzes revenue trends and patterns",
                "system_prompt": "You are a revenue analyst. Focus on revenue trends, growth patterns, and seasonal variations across product lines.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Cost Analyst",
                "description": "Analyzes cost structures and margins",
                "system_prompt": "You are a cost analyst. Examine cost structures, margin analysis, and expense categorization.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Performance Calculator",
                "description": "Calculates KPIs and financial metrics",
                "system_prompt": "You are a financial metrics specialist. Calculate KPIs, ratios, and performance indicators from the analyzed data.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.1
            },
            {
                "agent_name": "Report Generator",
                "description": "Creates structured financial reports",
                "system_prompt": "You are a report generator. Create comprehensive, well-structured financial reports with insights and recommendations.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.4
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
        print("SpreadSheetSwarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Structured analysis: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "financial-analysis-spreadsheet",
  "swarm_type": "SpreadSheetSwarm",
  "task": "Analyze quarterly financial performance data for a retail company with multiple product lines and create comprehensive insights",
  "output": {
    "data_validation": {
      "data_quality": "95% accuracy after cleaning",
      "missing_values": "Identified and filled 3% missing entries",
      "data_structure": "Standardized format across all product lines"
    },
    "revenue_analysis": {
      "q4_revenue": "$2.4M total revenue",
      "growth_rate": "12% quarter-over-quarter growth",
      "top_performers": ["Product Line A: +18%", "Product Line C: +15%"],
      "seasonal_trends": "Strong holiday season performance"
    },
    "cost_analysis": {
      "total_costs": "$1.8M operational costs",
      "cost_breakdown": "60% COGS, 25% Marketing, 15% Operations",
      "margin_analysis": "25% gross margin, 15% net margin",
      "cost_optimization": "Identified 8% potential savings in supply chain"
    },
    "performance_metrics": {
      "roi": "22% return on investment",
      "customer_acquisition_cost": "$45 per customer",
      "lifetime_value": "$320 average CLV",
      "inventory_turnover": "6.2x annual turnover"
    },
    "comprehensive_report": {
      "executive_summary": "Strong Q4 performance with 12% growth...",
      "recommendations": ["Expand Product Line A", "Optimize supply chain", "Increase marketing for underperformers"],
      "forecast": "Projected 15% growth for Q1 based on trends"
    }
  },
  "metadata": {
    "processing_structure": {
      "rows_processed": 1250,
      "columns_analyzed": 18,
      "calculations_performed": 47
    },
    "data_pipeline": [
      "Data Validation",
      "Revenue Analysis", 
      "Cost Analysis",
      "Performance Calculation",
      "Report Generation"
    ],
    "execution_time_seconds": 34.2,
    "billing_info": {
      "total_cost": 0.078
    }
  }
}
```

## Best Practices

- Structure data in clear, logical formats before processing
- Use systematic, step-by-step analysis approaches
- Ideal for quantitative analysis and reporting tasks
- Ensure data validation before proceeding with analysis

## Related Swarm Types

- [SequentialWorkflow](sequential_workflow.md) - For ordered data processing
- [ConcurrentWorkflow](concurrent_workflow.md) - For parallel data analysis
- [HierarchicalSwarm](hierarchical_swarm.md) - For complex data projects