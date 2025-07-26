# ModelRouter Docs

The ModelRouter is an intelligent routing system that automatically selects and executes AI models based on task requirements. It leverages a function-calling architecture to analyze tasks and recommend the optimal model and provider combination for each specific use case.





### Key Features

- Dynamic model selection based on task complexity and requirements
- Multi-provider support (OpenAI, Anthropic, Google, etc.)
- Concurrent and asynchronous execution capabilities
- Batch processing with memory
- Automatic error handling and retries
- Provider-aware routing
- Cost optimization

### Constructor Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| system_prompt | str | model_router_system_prompt | Custom prompt for guiding model selection behavior |
| max_tokens | int | 4000 | Maximum token limit for model outputs |
| temperature | float | 0.5 | Control parameter for response randomness (0.0-1.0) |
| max_workers | int/str | 10 | Maximum concurrent workers ("auto" for CPU count) |
| api_key | str | None | API key for model access |
| max_loops | int | 1 | Maximum number of refinement iterations |
| *args | Any | None | Additional positional arguments |
| **kwargs | Any | None | Additional keyword arguments |

### Core Methods

#### run(task: str) -> str

Executes a single task through the model router with memory and refinement capabilities.

# Installation

1. Install the latest version of swarms using pip:

```bash
pip3 install -U swarms
```

2. Setup your API Keys in your .env file with the following:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
# Add more API keys as needed following litellm format
```


```python
from swarms import ModelRouter

router = ModelRouter()

# Simple text analysis
result = router.run("Analyze the sentiment and key themes in this customer feedback")

# Complex reasoning task
complex_result = router.run("""
Evaluate the following business proposal:
- Initial investment: $500,000
- Projected ROI: 25% annually
- Market size: $2B
- Competition: 3 major players
Provide detailed analysis and recommendations.
""")
```

#### batch_run(tasks: list) -> list
Executes multiple tasks sequentially with result aggregation.

```python
# Multiple analysis tasks
tasks = [
    "Analyze Q1 financial performance",
    "Predict Q2 market trends",
    "Evaluate competitor strategies",
    "Generate growth recommendations"
]

results = router.batch_run(tasks)

# Process results
for task, result in zip(tasks, results):
    print(f"Task: {task}\nResult: {result}\n")
```

#### concurrent_run(tasks: list) -> list
Parallel execution of multiple tasks using thread pooling.

```python
import asyncio
from typing import List

# Define multiple concurrent tasks
analysis_tasks = [
    "Perform technical analysis of AAPL stock",
    "Analyze market sentiment from social media",
    "Generate trading signals",
    "Calculate risk metrics"
]

# Execute tasks concurrently
results = router.concurrent_run(analysis_tasks)

# Process results with error handling
for task, result in zip(analysis_tasks, results):
    try:
        processed_result = process_analysis(result)
        save_to_database(processed_result)
    except Exception as e:
        log_error(f"Error processing {task}: {str(e)}")
```

#### async_run(task: str) -> asyncio.Task
Asynchronous task execution with coroutine support.

```python
async def process_data_stream():
    tasks = []
    async for data in data_stream:
        task = await router.async_run(f"Process data: {data}")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# Usage in async context
async def main():
    router = ModelRouter()
    results = await process_data_stream()
```

### Advanced Usage Examples

#### Financial Analysis System

```python
from swarms import ModelRouter
from typing import Dict, List
import pandas as pd

class FinancialAnalysisSystem:
    def __init__(self):
        self.router = ModelRouter(
            temperature=0.3,  # Lower temperature for more deterministic outputs
            max_tokens=8000,  # Higher token limit for detailed analysis
            max_loops=2  # Allow for refinement iteration
        )
    
    def analyze_company_financials(self, financial_data: Dict) -> Dict:
        analysis_task = f"""
        Perform comprehensive financial analysis:
        
        Financial Metrics:
        - Revenue: ${financial_data['revenue']}M
        - EBITDA: ${financial_data['ebitda']}M
        - Debt/Equity: {financial_data['debt_equity']}
        - Working Capital: ${financial_data['working_capital']}M
        
        Required Analysis:
        1. Profitability assessment
        2. Liquidity analysis
        3. Growth projections
        4. Risk evaluation
        5. Investment recommendations
        
        Provide detailed insights and actionable recommendations.
        """
        
        result = self.router.run(analysis_task)
        return self._parse_analysis_result(result)
    
    def _parse_analysis_result(self, result: str) -> Dict:
        # Implementation of result parsing
        pass

# Usage
analyzer = FinancialAnalysisSystem()
company_data = {
    'revenue': 150,
    'ebitda': 45,
    'debt_equity': 0.8,
    'working_capital': 25
}

analysis = analyzer.analyze_company_financials(company_data)
```

#### Healthcare Data Processing Pipeline

```python
from swarms import ModelRouter
import pandas as pd
from typing import List, Dict

class MedicalDataProcessor:
    def __init__(self):
        self.router = ModelRouter(
            max_workers="auto",  # Automatic worker scaling
            temperature=0.2,     # Conservative temperature for medical analysis
            system_prompt="""You are a specialized medical data analyzer focused on:
            1. Clinical terminology interpretation
            2. Patient data analysis
            3. Treatment recommendation review
            4. Medical research synthesis"""
        )
    
    async def process_patient_records(self, records: List[Dict]) -> List[Dict]:
        analysis_tasks = []
        
        for record in records:
            task = f"""
            Analyze patient record:
            - Age: {record['age']}
            - Symptoms: {', '.join(record['symptoms'])}
            - Vital Signs: {record['vitals']}
            - Medications: {', '.join(record['medications'])}
            - Lab Results: {record['lab_results']}
            
            Provide:
            1. Symptom analysis
            2. Medication interaction check
            3. Lab results interpretation
            4. Treatment recommendations
            """
            analysis_tasks.append(task)
        
        results = await asyncio.gather(*[
            self.router.async_run(task) for task in analysis_tasks
        ])
        
        return [self._parse_medical_analysis(r) for r in results]
    
    def _parse_medical_analysis(self, analysis: str) -> Dict:
        # Implementation of medical analysis parsing
        pass

# Usage
async def main():
    processor = MedicalDataProcessor()
    patient_records = [
        {
            'age': 45,
            'symptoms': ['fever', 'cough', 'fatigue'],
            'vitals': {'bp': '120/80', 'temp': '38.5C'},
            'medications': ['lisinopril', 'metformin'],
            'lab_results': 'WBC: 11,000, CRP: 2.5'
        }
        # More records...
    ]
    
    analyses = await processor.process_patient_records(patient_records)
```

#### Natural Language Processing Pipeline

```python
from swarms import ModelRouter
from typing import List, Dict
import asyncio

class NLPPipeline:
    def __init__(self):
        self.router = ModelRouter(
            temperature=0.4,
            max_loops=2
        )
    
    def process_documents(self, documents: List[str]) -> List[Dict]:
        tasks = [self._create_nlp_task(doc) for doc in documents]
        results = self.router.concurrent_run(tasks)
        return [self._parse_nlp_result(r) for r in results]
    
    def _create_nlp_task(self, document: str) -> str:
        return f"""
        Perform comprehensive NLP analysis:
        
        Text: {document}
        
        Required Analysis:
        1. Entity recognition
        2. Sentiment analysis
        3. Topic classification
        4. Key phrase extraction
        5. Intent detection
        
        Provide structured analysis with confidence scores.
        """
    
    def _parse_nlp_result(self, result: str) -> Dict:
        # Implementation of NLP result parsing
        pass

# Usage
pipeline = NLPPipeline()
documents = [
    "We're extremely satisfied with the new product features!",
    "The customer service response time needs improvement.",
    "Looking to upgrade our subscription plan next month."
]

analyses = pipeline.process_documents(documents)
```

### Available Models and Use Cases

| Model | Provider | Optimal Use Cases | Characteristics |
|-------|----------|-------------------|-----------------|
| gpt-4-turbo | OpenAI | Complex reasoning, Code generation, Creative writing | High accuracy, Latest knowledge cutoff |
| claude-3-opus | Anthropic | Research analysis, Technical documentation, Long-form content | Strong reasoning, Detailed outputs |
| gemini-pro | Google | Multimodal tasks, Code generation, Technical analysis | Fast inference, Strong coding abilities |
| mistral-large | Mistral | General tasks, Content generation, Classification | Open source, Good price/performance |
| deepseek-reasoner | DeepSeek | Mathematical analysis, Logic problems, Scientific computing | Specialized reasoning capabilities |

### Provider Capabilities

| Provider | Strengths | Best For | Integration Notes |
|----------|-----------|-----------|------------------|
| OpenAI | Consistent performance, Strong reasoning | Production systems, Complex tasks | Requires API key setup |
| Anthropic | Safety features, Detailed analysis | Research, Technical writing | Claude-specific formatting |
| Google | Technical tasks, Multimodal support | Code generation, Analysis | Vertex AI integration available |
| Groq | High-speed inference | Real-time applications | Optimized for specific models |
| DeepSeek | Specialized reasoning | Scientific computing | Custom API integration |
| Mistral | Open source flexibility | General applications | Self-hosted options available |


### Performance Optimization Tips

1. Token Management
   - Set appropriate max_tokens based on task complexity
   - Monitor token usage for cost optimization
   - Use streaming for long outputs

2. Concurrency Settings
   - Adjust max_workers based on system resources
   - Use "auto" workers for optimal CPU utilization
   - Monitor memory usage with large batch sizes

3. Temperature Tuning
   - Lower (0.1-0.3) for factual/analytical tasks
   - Higher (0.7-0.9) for creative tasks
   - Mid-range (0.4-0.6) for balanced outputs

4. System Prompts
   - Customize for specific domains
   - Include relevant context
   - Define clear output formats

### Dependencies

- asyncio: Asynchronous I/O support
- concurrent.futures: Thread pool execution
- pydantic: Data validation
- litellm: LLM interface standardization
