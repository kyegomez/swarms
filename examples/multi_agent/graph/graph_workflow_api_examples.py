"""
GraphWorkflow API Examples

This file demonstrates how to use the Swarms API correctly with the proper format
and cheapest models for real-world GraphWorkflow scenarios.
"""

import os
import requests
import json
from typing import Dict, Any, List
from datetime import datetime

# API Configuration - Get API key from environment variable
API_KEY = os.getenv("SWARMS_API_KEY")
if not API_KEY:
    print("⚠️  Warning: SWARMS_API_KEY environment variable not set.")
    print("   Please set your API key: export SWARMS_API_KEY='your-api-key-here'")
    print("   Or set it in your environment variables.")
    API_KEY = "your-api-key-here"  # Placeholder for demonstration

BASE_URL = "https://api.swarms.world"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}


class SwarmsAPIExamples:
    """Examples of using Swarms API for GraphWorkflow scenarios."""
    
    def __init__(self):
        """Initialize API examples."""
        self.results = {}
        
    def health_check(self):
        """Check API health."""
        try:
            response = requests.get(f"{BASE_URL}/health", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def run_single_agent(self, task: str, agent_name: str = "Research Analyst"):
        """Run a single agent with the cheapest model."""
        payload = {
            "agent_config": {
                "agent_name": agent_name,
                "description": "An expert agent for various tasks",
                "system_prompt": (
                    "You are an expert assistant. Provide clear, concise, and accurate responses "
                    "to the given task. Focus on practical solutions and actionable insights."
                ),
                "model_name": "gpt-4o-mini",  # Cheapest model
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 4096,  # Reduced for cost
                "temperature": 0.7,
                "auto_generate_prompt": False,
                "tools_list_dictionary": None,
            },
            "task": task,
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/agent/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Single agent request failed: {e}")
            return None
    
    def run_sequential_swarm(self, task: str, agents: List[Dict[str, str]]):
        """Run a sequential swarm with multiple agents."""
        payload = {
            "name": "Sequential Workflow",
            "description": "Multi-agent sequential workflow",
            "agents": [
                {
                    "agent_name": agent["name"],
                    "description": agent["description"],
                    "system_prompt": agent["system_prompt"],
                    "model_name": "gpt-4o-mini",  # Cheapest model
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 4096,  # Reduced for cost
                    "temperature": 0.7,
                    "auto_generate_prompt": False
                }
                for agent in agents
            ],
            "max_loops": 1,
            "swarm_type": "SequentialWorkflow",
            "task": task
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/swarm/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Sequential swarm request failed: {e}")
            return None
    
    def run_concurrent_swarm(self, task: str, agents: List[Dict[str, str]]):
        """Run a concurrent swarm with multiple agents."""
        payload = {
            "name": "Concurrent Workflow",
            "description": "Multi-agent concurrent workflow",
            "agents": [
                {
                    "agent_name": agent["name"],
                    "description": agent["description"],
                    "system_prompt": agent["system_prompt"],
                    "model_name": "gpt-4o-mini",  # Cheapest model
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 4096,  # Reduced for cost
                    "temperature": 0.7,
                    "auto_generate_prompt": False
                }
                for agent in agents
            ],
            "max_loops": 1,
            "swarm_type": "ConcurrentWorkflow",
            "task": task
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/swarm/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Concurrent swarm request failed: {e}")
            return None
    
    def example_software_development_pipeline(self):
        """Example: Software Development Pipeline using Swarms API."""
        print("\n🔧 Example: Software Development Pipeline")
        print("-" * 50)
        
        # Define agents for software development
        agents = [
            {
                "name": "CodeGenerator",
                "description": "Generates clean, well-documented code",
                "system_prompt": "You are an expert Python developer. Generate clean, well-documented code with proper error handling and documentation."
            },
            {
                "name": "CodeReviewer",
                "description": "Reviews code for bugs and best practices",
                "system_prompt": "You are a senior code reviewer. Check for bugs, security issues, and best practices. Provide specific feedback and suggestions."
            },
            {
                "name": "TestGenerator",
                "description": "Generates comprehensive unit tests",
                "system_prompt": "You are a QA engineer. Generate comprehensive unit tests for the given code with good coverage and edge cases."
            }
        ]
        
        task = "Create a Python function that implements a binary search algorithm with proper error handling and documentation"
        
        result = self.run_sequential_swarm(task, agents)
        if result:
            print("✅ Software Development Pipeline completed successfully")
            # Debug: Print the full response structure
            print(f"🔍 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            # Try different possible result keys
            result_text = (
                result.get('result') or 
                result.get('response') or 
                result.get('content') or 
                result.get('output') or 
                result.get('data') or 
                str(result)[:200]
            )
            print(f"📝 Result: {result_text[:200] if result_text else 'No result'}...")
        else:
            print("❌ Software Development Pipeline failed")
        
        return result
    
    def example_data_analysis_pipeline(self):
        """Example: Data Analysis Pipeline using Swarms API."""
        print("\n📊 Example: Data Analysis Pipeline")
        print("-" * 50)
        
        # Define agents for data analysis
        agents = [
            {
                "name": "DataExplorer",
                "description": "Explores and analyzes data patterns",
                "system_prompt": "You are a data scientist. Analyze the given data, identify patterns, trends, and key insights. Provide clear explanations."
            },
            {
                "name": "StatisticalAnalyst",
                "description": "Performs statistical analysis",
                "system_prompt": "You are a statistical analyst. Perform statistical analysis on the data, identify correlations, and provide statistical insights."
            },
            {
                "name": "ReportWriter",
                "description": "Creates comprehensive reports",
                "system_prompt": "You are a report writer. Create comprehensive, well-structured reports based on the analysis. Include executive summaries and actionable recommendations."
            }
        ]
        
        task = "Analyze this customer transaction data and provide insights on purchasing patterns, customer segments, and recommendations for business growth"
        
        result = self.run_sequential_swarm(task, agents)
        if result:
            print("✅ Data Analysis Pipeline completed successfully")
            # Try different possible result keys
            result_text = (
                result.get('result') or 
                result.get('response') or 
                result.get('content') or 
                result.get('output') or 
                result.get('data') or 
                str(result)[:200]
            )
            print(f"📝 Result: {result_text[:200] if result_text else 'No result'}...")
        else:
            print("❌ Data Analysis Pipeline failed")
        
        return result
    
    def example_business_process_workflow(self):
        """Example: Business Process Workflow using Swarms API."""
        print("\n💼 Example: Business Process Workflow")
        print("-" * 50)
        
        # Define agents for business process
        agents = [
            {
                "name": "BusinessAnalyst",
                "description": "Analyzes business requirements and processes",
                "system_prompt": "You are a business analyst. Analyze business requirements, identify process improvements, and provide strategic recommendations."
            },
            {
                "name": "ProcessDesigner",
                "description": "Designs optimized business processes",
                "system_prompt": "You are a process designer. Design optimized business processes based on analysis, considering efficiency, cost, and scalability."
            },
            {
                "name": "ImplementationPlanner",
                "description": "Plans implementation strategies",
                "system_prompt": "You are an implementation planner. Create detailed implementation plans, timelines, and resource requirements for process changes."
            }
        ]
        
        task = "Analyze our current customer onboarding process and design an optimized workflow that reduces time-to-value while maintaining quality"
        
        result = self.run_sequential_swarm(task, agents)
        if result:
            print("✅ Business Process Workflow completed successfully")
            # Try different possible result keys
            result_text = (
                result.get('result') or 
                result.get('response') or 
                result.get('content') or 
                result.get('output') or 
                result.get('data') or 
                str(result)[:200]
            )
            print(f"📝 Result: {result_text[:200] if result_text else 'No result'}...")
        else:
            print("❌ Business Process Workflow failed")
        
        return result
    
    def example_concurrent_research(self):
        """Example: Concurrent Research using Swarms API."""
        print("\n🔍 Example: Concurrent Research")
        print("-" * 50)
        
        # Define agents for concurrent research
        agents = [
            {
                "name": "MarketResearcher",
                "description": "Researches market trends and competition",
                "system_prompt": "You are a market researcher. Research market trends, competitive landscape, and industry developments. Focus on actionable insights."
            },
            {
                "name": "TechnologyAnalyst",
                "description": "Analyzes technology trends and innovations",
                "system_prompt": "You are a technology analyst. Research technology trends, innovations, and emerging technologies. Provide technical insights and predictions."
            },
            {
                "name": "FinancialAnalyst",
                "description": "Analyzes financial data and market performance",
                "system_prompt": "You are a financial analyst. Analyze financial data, market performance, and economic indicators. Provide financial insights and forecasts."
            }
        ]
        
        task = "Research the current state of artificial intelligence in healthcare, including market size, key players, technological advances, and future opportunities"
        
        result = self.run_concurrent_swarm(task, agents)
        if result:
            print("✅ Concurrent Research completed successfully")
            # Try different possible result keys
            result_text = (
                result.get('result') or 
                result.get('response') or 
                result.get('content') or 
                result.get('output') or 
                result.get('data') or 
                str(result)[:200]
            )
            print(f"📝 Result: {result_text[:200] if result_text else 'No result'}...")
        else:
            print("❌ Concurrent Research failed")
        
        return result
    
    def run_all_examples(self):
        """Run all API examples."""
        print("🚀 Starting Swarms API Examples")
        print("=" * 60)
        
        # Check API health first
        print("\n🔍 Checking API Health...")
        health = self.health_check()
        if health:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return
        
        # Run examples
        examples = [
            self.example_software_development_pipeline,
            self.example_data_analysis_pipeline,
            self.example_business_process_workflow,
            self.example_concurrent_research,
        ]
        
        for example in examples:
            try:
                result = example()
                if result:
                    self.results[example.__name__] = result
            except Exception as e:
                print(f"❌ Example {example.__name__} failed: {e}")
                self.results[example.__name__] = {"error": str(e)}
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate a summary of all examples."""
        print("\n" + "=" * 60)
        print("📊 SWARMS API EXAMPLES SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for result in self.results.values() if "error" not in result)
        failed = len(self.results) - successful
        
        print(f"Total Examples: {len(self.results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        print("\n📈 Results:")
        print("-" * 60)
        
        for name, result in self.results.items():
            if "error" in result:
                print(f"❌ {name}: {result['error']}")
            else:
                print(f"✅ {name}: Completed successfully")
        
        # Save results to file
        report_data = {
            "summary": {
                "total_examples": len(self.results),
                "successful": successful,
                "failed": failed,
                "timestamp": datetime.now().isoformat()
            },
            "results": self.results
        }
        
        with open("swarms_api_examples_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: swarms_api_examples_report.json")


def main():
    """Main function to run all API examples."""
    examples = SwarmsAPIExamples()
    results = examples.run_all_examples()
    return results


if __name__ == "__main__":
    # Run API examples
    main() 