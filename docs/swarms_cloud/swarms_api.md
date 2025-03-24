# Swarms API Documentation

*Enterprise-grade Agent Swarm Management API*

**Base URL**: `https://api.swarms.world`  
**API Key Management**: [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)  

## Overview

The Swarms API provides a robust, scalable infrastructure for deploying and managing intelligent agent swarms in the cloud. This enterprise-grade API enables organizations to create, execute, and orchestrate sophisticated AI agent workflows without managing the underlying infrastructure.

Key capabilities include:

- **Intelligent Swarm Management**: Create and execute swarms of specialized AI agents that collaborate to solve complex tasks
- **Automatic Agent Generation**: Dynamically create optimized agents based on task requirements
- **Multiple Swarm Architectures**: Choose from various swarm patterns to match your specific workflow needs
- **Scheduled Execution**: Set up automated, scheduled swarm executions
- **Comprehensive Logging**: Track and analyze all API interactions
- **Cost Management**: Predictable, transparent pricing with optimized resource utilization
- **Enterprise Security**: Full API key authentication and management

Swarms API is designed for production use cases requiring sophisticated AI orchestration, with applications in finance, healthcare, legal, research, and other domains where complex reasoning and multi-agent collaboration are needed.

## Authentication

All API requests require a valid API key, which must be included in the header of each request:

```
x-api-key: your_api_key_here
```

API keys can be obtained and managed at [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys).

## API Reference

### Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Simple health check endpoint |
| `/v1/swarm/completions` | POST | Run a swarm with specified configuration |
| `/v1/swarm/batch/completions` | POST | Run multiple swarms in batch mode |
| `/v1/swarm/schedule` | POST | Schedule a swarm to run at a specific time |
| `/v1/swarm/schedule` | GET | Get all scheduled swarm jobs |
| `/v1/swarm/schedule/{job_id}` | DELETE | Cancel a scheduled swarm job |
| `/v1/swarm/logs` | GET | Retrieve API request logs |
| `/v1/swarms/available` | GET | Get all available swarms as a list of strings |
| `/v1/models/available` | GET | Get all available models as a list of strings |



### SwarmType Reference

The `swarm_type` parameter defines the architecture and collaboration pattern of the agent swarm:

| SwarmType | Description |
|-----------|-------------|
| `AgentRearrange` | Dynamically reorganizes the workflow between agents based on task requirements |
| `MixtureOfAgents` | Combines multiple agent types to tackle diverse aspects of a problem |
| `SpreadSheetSwarm` | Specialized for spreadsheet data analysis and manipulation |
| `SequentialWorkflow` | Agents work in a predefined sequence, each handling specific subtasks |
| `ConcurrentWorkflow` | Multiple agents work simultaneously on different aspects of the task |
| `GroupChat` | Agents collaborate in a discussion format to solve problems |
| `MultiAgentRouter` | Routes subtasks to specialized agents based on their capabilities |
| `AutoSwarmBuilder` | Automatically designs and builds an optimal swarm based on the task |
| `HiearchicalSwarm` | Organizes agents in a hierarchical structure with managers and workers |
| `MajorityVoting` | Uses a consensus mechanism where multiple agents vote on the best solution |
| `auto` | Automatically selects the most appropriate swarm type for the given task |



## Data Models

### SwarmSpec

The `SwarmSpec` model defines the configuration of a swarm.

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| name | string | Identifier for the swarm | No |
| description | string | Description of the swarm's purpose | No |
| agents | Array<AgentSpec> | List of agent specifications | No |
| max_loops | integer | Maximum number of execution loops | No |
| swarm_type | SwarmType | Architecture of the swarm | No |
| rearrange_flow | string | Instructions for rearranging task flow | No |
| task | string | The main task for the swarm to accomplish | Yes |
| img | string | Optional image URL for the swarm | No |
| return_history | boolean | Whether to return execution history | No |
| rules | string | Guidelines for swarm behavior | No |
| schedule | ScheduleSpec | Scheduling information | No |

### AgentSpec

The `AgentSpec` model defines the configuration of an individual agent.

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| agent_name | string | Unique name for the agent | Yes* |
| description | string | Description of the agent's purpose | No |
| system_prompt | string | Instructions for the agent | No |
| model_name | string | AI model to use (e.g., "gpt-4o") | Yes* |
| auto_generate_prompt | boolean | Whether to auto-generate prompts | No |
| max_tokens | integer | Maximum tokens in response | No |
| temperature | float | Randomness of responses (0-1) | No |
| role | string | Agent's role in the swarm | No |
| max_loops | integer | Maximum iterations for this agent | No |

*Required if agents are manually specified; not required if using auto-generated agents

### ScheduleSpec

The `ScheduleSpec` model defines when a swarm should be executed.

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| scheduled_time | datetime | Time when the swarm should run | Yes |
| timezone | string | Timezone for the scheduled time | No (defaults to "UTC") |



### Endpoint Details

#### Health Check

Check if the API service is available and functioning correctly.

**Endpoint**: `/health`  
**Method**: GET  
**Rate Limit**: 100 requests per 60 seconds

**Example Request**:
```bash
curl -X GET "https://api.swarms.world/health" \
     -H "x-api-key: your_api_key_here"
```

**Example Response**:
```json
{
  "status": "ok"
}
```

#### Run Swarm

Run a swarm with the specified configuration to complete a task.

**Endpoint**: `/v1/swarm/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Parameters**:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| name | string | Identifier for the swarm | No |
| description | string | Description of the swarm's purpose | No |
| agents | Array<AgentSpec> | List of agent specifications | No |
| max_loops | integer | Maximum number of execution loops | No |
| swarm_type | SwarmType | Architecture of the swarm | No |
| rearrange_flow | string | Instructions for rearranging task flow | No |
| task | string | The main task for the swarm to accomplish | Yes |
| img | string | Optional image URL for the swarm | No |
| return_history | boolean | Whether to return execution history | No |
| rules | string | Guidelines for swarm behavior | No |
| schedule | ScheduleSpec | Scheduling information | No |

**Example Request**:
```bash

# Run single swarm
curl -X POST "https://api.swarms.world/v1/swarm/completions" \
  -H "x-api-key: $SWARMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Financial Analysis Swarm",
    "description": "Market analysis swarm",
    "agents": [
      {
        "agent_name": "Market Analyst",
        "description": "Analyzes market trends",
        "system_prompt": "You are a financial analyst expert.",
        "model_name": "openai/gpt-4o",
        "role": "worker",
        "max_loops": 1,
        "max_tokens": 8192,
        "temperature": 0.5,
        "auto_generate_prompt": false
      },
      {
        "agent_name": "Economic Forecaster",
        "description": "Predicts economic trends",
        "system_prompt": "You are an expert in economic forecasting.",
        "model_name": "gpt-4o",
        "role": "worker",
        "max_loops": 1,
        "max_tokens": 8192,
        "temperature": 0.5,
        "auto_generate_prompt": false
      }
    ],
    "max_loops": 1,
    "swarm_type": "ConcurrentWorkflow",
    "task": "What are the best etfs and index funds for ai and tech?",
    "output_type": "dict"
  }'

```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "financial-analysis-swarm",
  "description": "Analyzes financial data for risk assessment",
  "swarm_type": "SequentialWorkflow",
  "task": "Analyze the provided quarterly financials for Company XYZ and identify potential risk factors. Summarize key insights and provide recommendations for risk mitigation.",
  "output": {
    "financial_analysis": {
      "risk_factors": [...],
      "key_insights": [...],
      "recommendations": [...]
    }
  },
  "metadata": {
    "max_loops": 2,
    "num_agents": 3,
    "execution_time_seconds": 12.45,
    "completion_time": 1709563245.789,
    "billing_info": {
      "cost_breakdown": {
        "agent_cost": 0.03,
        "input_token_cost": 0.002134,
        "output_token_cost": 0.006789,
        "token_counts": {
          "total_input_tokens": 1578,
          "total_output_tokens": 3456,
          "total_tokens": 5034,
          "per_agent": {...}
        },
        "num_agents": 3,
        "execution_time_seconds": 12.45
      },
      "total_cost": 0.038923
    }
  }
}
```

#### Run Batch Completions

Run multiple swarms as a batch operation.

**Endpoint**: `/v1/swarm/batch/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Parameters**:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| swarms | Array<SwarmSpec> | List of swarm specifications | Yes |

**Example Request**:
```bash
# Batch swarm completions
curl -X POST "https://api.swarms.world/v1/swarm/batch/completions" \
  -H "x-api-key: $SWARMS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "name": "Batch Swarm 1",
      "description": "First swarm in the batch",
      "agents": [
        {
          "agent_name": "Research Agent",
          "description": "Conducts research",
          "system_prompt": "You are a research assistant.",
          "model_name": "gpt-4o",
          "role": "worker",
          "max_loops": 1
        },
        {
          "agent_name": "Analysis Agent",
          "description": "Analyzes data",
          "system_prompt": "You are a data analyst.",
          "model_name": "gpt-4o",
          "role": "worker",
          "max_loops": 1
        }
      ],
      "max_loops": 1,
      "swarm_type": "SequentialWorkflow",
      "task": "Research AI advancements."
    },
    {
      "name": "Batch Swarm 2",
      "description": "Second swarm in the batch",
      "agents": [
        {
          "agent_name": "Writing Agent",
          "description": "Writes content",
          "system_prompt": "You are a content writer.",
          "model_name": "gpt-4o",
          "role": "worker",
          "max_loops": 1
        },
        {
          "agent_name": "Editing Agent",
          "description": "Edits content",
          "system_prompt": "You are an editor.",
          "model_name": "gpt-4o",
          "role": "worker",
          "max_loops": 1
        }
      ],
      "max_loops": 1,
      "swarm_type": "SequentialWorkflow",
      "task": "Write a summary of AI research."
    }
  ]'
```

**Example Response**:
```json
[
  {
    "status": "success",
    "swarm_name": "risk-analysis",
    "task": "Analyze risk factors for investment portfolio",
    "output": {...},
    "metadata": {...}
  },
  {
    "status": "success",
    "swarm_name": "market-sentiment",
    "task": "Assess current market sentiment for technology sector",
    "output": {...},
    "metadata": {...}
  }
]
```

#### Schedule Swarm

Schedule a swarm to run at a specific time.

**Endpoint**: `/v1/swarm/schedule`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Parameters**:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| name | string | Identifier for the swarm | No |
| description | string | Description of the swarm's purpose | No |
| agents | Array<AgentSpec> | List of agent specifications | No |
| max_loops | integer | Maximum number of execution loops | No |
| swarm_type | SwarmType | Architecture of the swarm | No |
| task | string | The main task for the swarm to accomplish | Yes |
| schedule | ScheduleSpec | Scheduling information | Yes |

**Example Request**:
```bash
curl -X POST "https://api.swarms.world/v1/swarm/schedule" \
     -H "x-api-key: your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "daily-market-analysis",
       "description": "Daily analysis of market conditions",
       "task": "Analyze today's market movements and prepare a summary report",
       "schedule": {
         "scheduled_time": "2025-03-05T17:00:00Z",
         "timezone": "UTC"
       }
     }'
```

**Example Response**:
```json
{
  "status": "success",
  "message": "Swarm scheduled successfully",
  "job_id": "swarm_daily-market-analysis_1709563245",
  "scheduled_time": "2025-03-05T17:00:00Z",
  "timezone": "UTC"
}
```

#### Get Scheduled Jobs

Retrieve all scheduled swarm jobs.

**Endpoint**: `/v1/swarm/schedule`  
**Method**: GET  
**Rate Limit**: 100 requests per 60 seconds

**Example Request**:
```bash
curl -X GET "https://api.swarms.world/v1/swarm/schedule" \
     -H "x-api-key: your_api_key_here"
```

**Example Response**:
```json
{
  "status": "success",
  "scheduled_jobs": [
    {
      "job_id": "swarm_daily-market-analysis_1709563245",
      "swarm_name": "daily-market-analysis",
      "scheduled_time": "2025-03-05T17:00:00Z",
      "timezone": "UTC"
    },
    {
      "job_id": "swarm_weekly-report_1709563348",
      "swarm_name": "weekly-report",
      "scheduled_time": "2025-03-09T12:00:00Z",
      "timezone": "UTC"
    }
  ]
}
```

#### Cancel Scheduled Job

Cancel a previously scheduled swarm job.

**Endpoint**: `/v1/swarm/schedule/{job_id}`  
**Method**: DELETE  
**Rate Limit**: 100 requests per 60 seconds

**Path Parameters**:

| Parameter | Description |
|-----------|-------------|
| job_id | ID of the scheduled job to cancel |

**Example Request**:
```bash
curl -X DELETE "https://api.swarms.world/v1/swarm/schedule/swarm_daily-market-analysis_1709563245" \
     -H "x-api-key: your_api_key_here"
```

**Example Response**:
```json
{
  "status": "success",
  "message": "Scheduled job cancelled successfully",
  "job_id": "swarm_daily-market-analysis_1709563245"
}
```


### Get Models

#### Get Available Models

Get all available models as a list of strings.

**Endpoint**: `/v1/models/available`  
**Method**: GET  

**Example Request**:
```bash
curl -X GET "https://api.swarms.world/v1/models/available" \
     -H "x-api-key: your_api_key_here"
```


------


### Get Swarms Available

Get all available swarms as a list of strings.

**Endpoint**: `/v1/swarms/available`  
**Method**: GET

**Example Request**:
```bash
curl -X GET "https://api.swarms.world/v1/swarms/available" \
     -H "x-api-key: your_api_key_here"
```

**Example Response**:
```json
{
  "status": "success",
  "swarms": ["financial-analysis-swarm", "market-sentiment-swarm"]
}
```

-------


#### Get API Logs

Retrieve logs of API requests made with your API key.

**Endpoint**: `/v1/swarm/logs`  
**Method**: GET  
**Rate Limit**: 100 requests per 60 seconds

**Example Request**:
```bash
curl -X GET "https://api.swarms.world/v1/swarm/logs" \
     -H "x-api-key: your_api_key_here"
```

**Example Response**:
```json
{
  "status": "success",
  "count": 25,
  "logs": [
    {
      "id": "log_id_12345",
      "api_key": "api_key_redacted",
      "data": {
        "action": "run_swarm",
        "swarm_name": "financial-analysis-swarm",
        "task": "Analyze quarterly financials...",
        "timestamp": "2025-03-04T14:22:45Z"
      }
    },
    ...
  ]
}
```

## Production Examples

### Python Examples

#### Financial Risk Assessment (Python)

This example demonstrates creating a swarm for comprehensive financial risk assessment.

```python
import requests
import json
from datetime import datetime, timedelta

# API Configuration
API_BASE_URL = "https://api.swarms.world"
API_KEY = "your_api_key_here"
HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def financial_risk_assessment(company_data, market_conditions, risk_tolerance):
    """
    Creates and runs a swarm to perform comprehensive financial risk assessment.
    
    Args:
        company_data (str): Description or data about the company
        market_conditions (str): Current market conditions
        risk_tolerance (str): Risk tolerance level (e.g., "conservative", "moderate", "aggressive")
        
    Returns:
        dict: Risk assessment results
    """
    # Prepare the task description with all relevant information
    task = f"""
    Perform a comprehensive financial risk assessment with the following data:
    
    COMPANY DATA:
    {company_data}
    
    MARKET CONDITIONS:
    {market_conditions}
    
    RISK TOLERANCE:
    {risk_tolerance}
    
    Analyze all potential risk factors including market risks, credit risks, 
    operational risks, and regulatory compliance risks. Quantify each risk factor 
    on a scale of 1-10 and provide specific mitigation strategies.
    
    Return a detailed report with executive summary, risk scores, detailed analysis,
    and actionable recommendations.
    """
    
    # Define specialized financial agents
    financial_analysts = [
        {
            "agent_name": "MarketAnalyst",
            "description": "Specialist in market risk assessment and forecasting",
            "system_prompt": "You are an expert market analyst with deep expertise in financial markets. Analyze market conditions, trends, and external factors that could impact financial performance. Provide quantitative and qualitative analysis of market-related risks.",
            "model_name": "gpt-4o",
            "temperature": 0.3,
            "role": "analyst",
            "max_loops": 1
        },
        {
            "agent_name": "CreditRiskAnalyst",
            "description": "Expert in assessing credit and counterparty risks",
            "system_prompt": "You are a specialist in credit risk analysis with experience in banking and financial institutions. Evaluate creditworthiness, default probabilities, and counterparty exposures. Provide detailed analysis of credit-related risks and recommended safeguards.",
            "model_name": "gpt-4o",
            "temperature": 0.2,
            "role": "analyst",
            "max_loops": 1
        },
        {
            "agent_name": "RegulatoryExpert",
            "description": "Expert in financial regulations and compliance",
            "system_prompt": "You are a regulatory compliance expert with deep knowledge of financial regulations. Identify potential regulatory risks, compliance issues, and governance concerns. Recommend compliance measures and risk mitigation strategies.",
            "model_name": "gpt-4o",
            "temperature": 0.2,
            "role": "analyst",
            "max_loops": 1
        },
        {
            "agent_name": "RiskSynthesizer",
            "description": "Integrates all risk factors into comprehensive assessment",
            "system_prompt": "You are a senior risk management professional responsible for synthesizing multiple risk analyses into a coherent, comprehensive risk assessment. Integrate analyses from various domains, resolve conflicting assessments, and provide a holistic view of risk exposure with prioritized recommendations.",
            "model_name": "gpt-4o",
            "temperature": 0.4,
            "role": "manager",
            "max_loops": 1
        }
    ]
    
    # Create the swarm specification
    swarm_spec = {
        "name": "financial-risk-assessment",
        "description": "Comprehensive financial risk assessment swarm",
        "agents": financial_analysts,
        "max_loops": 2,
        "swarm_type": "HiearchicalSwarm",
        "task": task,
        "return_history": True
    }
    
    # Execute the swarm
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=HEADERS,
        json=swarm_spec
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Risk assessment completed. Cost: ${result['metadata']['billing_info']['total_cost']}")
        return result["output"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage example
if __name__ == "__main__":
    company_data = """
    XYZ Financial Services
    Annual Revenue: $125M
    Current Debt: $45M
    Credit Rating: BBB+
    Primary Markets: North America, Europe
    Key Products: Asset management, retirement planning, commercial lending
    Recent Events: Expanding into Asian markets, New CEO appointed 6 months ago
    """
    
    market_conditions = """
    Current interest rates rising (Federal Reserve increased rates by 0.25% last month)
    Inflation at 3.2% (12-month outlook projects 3.5-4.0%)
    Market volatility index (VIX) at 22.4 (elevated)
    Regulatory environment: New financial reporting requirements taking effect next quarter
    Sector performance: Financial services sector underperforming broader market by 2.7%
    """
    
    risk_tolerance = "moderate"
    
    result = financial_risk_assessment(company_data, market_conditions, risk_tolerance)
    
    if result:
        # Process and use the risk assessment
        print(json.dumps(result, indent=2))
        
        # Optionally, schedule a follow-up assessment
        tomorrow = datetime.utcnow() + timedelta(days=30)
        schedule_spec = {
            "name": "monthly-risk-update",
            "description": "Monthly update to risk assessment",
            "task": f"Update the risk assessment for XYZ Financial Services based on current market conditions. Previous assessment: {json.dumps(result)}",
            "schedule": {
                "scheduled_time": tomorrow.isoformat() + "Z",
                "timezone": "UTC"
            }
        }
        
        schedule_response = requests.post(
            f"{API_BASE_URL}/v1/swarm/schedule",
            headers=HEADERS,
            json=schedule_spec
        )
        
        if schedule_response.status_code == 200:
            print("Follow-up assessment scheduled successfully")
            print(schedule_response.json())
```

#### Healthcare Patient Data Analysis (Python)

This example demonstrates creating a swarm for analyzing patient health data and generating insights.

```python
import requests
import json
import os
from datetime import datetime

# API Configuration
API_BASE_URL = "https://api.swarms.world"
API_KEY = os.environ.get("SWARMS_API_KEY")
HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def analyze_patient_health_data(patient_data, medical_history, lab_results, treatment_goals):
    """
    Creates and runs a swarm to analyze patient health data and generate insights.
    
    Args:
        patient_data (str): Basic patient information
        medical_history (str): Patient's medical history
        lab_results (str): Recent laboratory results
        treatment_goals (str): Treatment objectives
        
    Returns:
        dict: Comprehensive health analysis and recommendations
    """
    # Prepare the detailed task description
    task = f"""
    Perform a comprehensive analysis of the following patient health data:
    
    PATIENT INFORMATION:
    {patient_data}
    
    MEDICAL HISTORY:
    {medical_history}
    
    LABORATORY RESULTS:
    {lab_results}
    
    TREATMENT GOALS:
    {treatment_goals}
    
    Analyze all aspects of the patient's health status, identify potential concerns,
    evaluate treatment effectiveness, and provide evidence-based recommendations for
    optimizing care. Consider medication interactions, lifestyle factors, and preventive measures.
    
    Return a detailed clinical report with key findings, risk stratification, 
    prioritized recommendations, and suggested follow-up timeline.
    """
    
    # Create the swarm specification with auto-generated agents
    # (letting the system create specialized medical experts)
    swarm_spec = {
        "name": "patient-health-analysis",
        "description": "Comprehensive patient health data analysis",
        "swarm_type": "AutoSwarmBuilder",
        "task": task,
        "max_loops": 3,
        "return_history": True
    }
    
    # Execute the swarm
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/swarm/completions",
            headers=HEADERS,
            json=swarm_spec
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Log the execution metadata
        execution_time = result["metadata"]["execution_time_seconds"]
        cost = result["metadata"]["billing_info"]["total_cost"]
        num_agents = result["metadata"]["num_agents"]
        
        print(f"Analysis completed in {execution_time:.2f} seconds")
        print(f"Used {num_agents} specialized medical agents")
        print(f"Total cost: ${cost:.4f}")
        
        # Return just the analysis results
        return result["output"]
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    # Sample patient data (would typically come from EHR system)
    patient_data = """
    ID: PT-28456
    Age: 67
    Gender: Female
    Height: 162 cm
    Weight: 78 kg
    Vitals:
    - Blood Pressure: 142/88 mmHg
    - Heart Rate: 76 bpm
    - Respiratory Rate: 16/min
    - Temperature: 37.1°C
    - Oxygen Saturation: 97%
    """
    
    medical_history = """
    Diagnoses:
    - Type 2 Diabetes Mellitus (diagnosed 12 years ago)
    - Hypertension (diagnosed 8 years ago)
    - Osteoarthritis (knees, diagnosed 5 years ago)
    - Hyperlipidemia
    
    Surgical History:
    - Cholecystectomy (15 years ago)
    - Right knee arthroscopy (3 years ago)
    
    Medications:
    - Metformin 1000mg BID
    - Lisinopril 20mg daily
    - Atorvastatin 40mg daily
    - Aspirin 81mg daily
    - Acetaminophen 500mg PRN for joint pain
    
    Allergies:
    - Penicillin (rash)
    - Sulfa drugs (hives)
    
    Family History:
    - Father: MI at age 70, died at 76
    - Mother: Breast cancer at 68, Type 2 Diabetes, died at 82
    - Sister: Type 2 Diabetes, Hypertension
    """
    
    lab_results = """
    CBC (2 days ago):
    - WBC: 7.2 x10^9/L (normal)
    - RBC: 4.1 x10^12/L (low-normal)
    - Hemoglobin: 12.8 g/dL (low-normal)
    - Hematocrit: 38% (low-normal)
    - Platelets: 245 x10^9/L (normal)
    
    Comprehensive Metabolic Panel:
    - Glucose (fasting): 142 mg/dL (elevated)
    - HbA1c: 7.8% (elevated)
    - BUN: 22 mg/dL (normal)
    - Creatinine: 1.1 mg/dL (normal)
    - eGFR: 62 mL/min/1.73m² (mildly reduced)
    - Sodium: 138 mEq/L (normal)
    - Potassium: 4.2 mEq/L (normal)
    - Chloride: 101 mEq/L (normal)
    - Calcium: 9.4 mg/dL (normal)
    - ALT: 32 U/L (normal)
    - AST: 28 U/L (normal)
    
    Lipid Panel:
    - Total Cholesterol: 198 mg/dL
    - Triglycerides: 172 mg/dL (elevated)
    - HDL: 42 mg/dL (low)
    - LDL: 122 mg/dL (borderline elevated)
    
    Urinalysis:
    - Microalbumin/Creatinine ratio: 45 mg/g (elevated)
    """
    
    treatment_goals = """
    Primary Goals:
    - Improve glycemic control (target HbA1c < 7.0%)
    - Blood pressure control (target < 130/80 mmHg)
    - Lipid management (target LDL < 100 mg/dL)
    - Renal protection (reduce microalbuminuria)
    - Weight management (target BMI < 27)
    - Pain management for osteoarthritis
    - Maintain functional independence
    
    Patient Preferences:
    - Prefers to minimize medication changes if possible
    - Interested in dietary approaches
    - Concerned about memory changes
    - Limited exercise tolerance due to knee pain
    """
    
    result = analyze_patient_health_data(patient_data, medical_history, lab_results, treatment_goals)
    
    if result:
        # Write the analysis to a report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"patient_analysis_{timestamp}.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Analysis saved to patient_analysis_{timestamp}.json")
        
        # Display key findings
        if "key_findings" in result:
            print("\nKEY FINDINGS:")
            for i, finding in enumerate(result["key_findings"]):
                print(f"  {i+1}. {finding}")
                
        # Display recommendations
        if "recommendations" in result:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(result["recommendations"]):
                print(f"  {i+1}. {rec}")
```

## Error Handling

The Swarms API follows standard HTTP status codes for error responses:

| Status Code | Meaning | Handling Strategy |
|-------------|---------|-------------------|
| 400 | Bad Request | Validate request parameters before sending |
| 401 | Unauthorized | Check API key validity |
| 403 | Forbidden | Verify API key permissions |
| 404 | Not Found | Check endpoint URL and resource IDs |
| 429 | Too Many Requests | Implement exponential backoff retry logic |
| 500 | Internal Server Error | Retry with backoff, then contact support |

Error responses include a detailed message explaining the issue:

```json
{
  "detail": "Failed to create swarm: Invalid swarm_type specified"
}
```

## Rate Limiting

The API enforces a rate limit of 100 requests per 60-second window. When exceeded, a 429 status code is returned. Implement appropriate retry logic with exponential backoff in production applications.

## Billing & Cost Management

The API uses a credit-based billing system with costs calculated based on:

1. **Agent Count**: Base cost per agent


2. **Input Tokens**: Cost based on the size of input data and prompts

3. **Output Tokens**: Cost based on the length of generated responses

4. **Time of Day**: Reduced rates during nighttime hours (8 PM to 6 AM PT)

Cost information is included in each response's metadata for transparency and forecasting.

## Best Practices

1. **Task Description**

   - Provide detailed, specific task descriptions

   - Include all necessary context and constraints
   
   - Structure complex inputs for easier processing

2. **Agent Configuration**
   
   - For simple tasks, use `AutoSwarmBuilder` to automatically generate optimal agents
   
   - For complex or specialized tasks, manually define agents with specific expertise
   
   - Use appropriate `swarm_type` for your workflow pattern

3. **Production Implementation**

  - Implement robust error handling and retries
  
  - Log API responses for debugging and auditing
  
  - Monitor costs closely during development and testing
  
  - Use scheduled jobs for recurring tasks instead of continuous polling

4. **Cost Optimization**

  - Batch related tasks when possible

  - Schedule non-urgent tasks during discount hours

  - Carefully scope task descriptions to reduce token usage

  - Cache results when appropriate


## Support

For technical assistance with the Swarms API, please contact:

- Documentation: [https://docs.swarms.world](https://docs.swarms.world)
- Email: kye@swarms.world
- Community Discord: [https://discord.gg/swarms](https://discord.gg/swarms)
- Swarms Marketplace: [https://swarms.world](https://swarms.world)
- Swarms AI Website: [https://swarms.ai](https://swarms.ai)
