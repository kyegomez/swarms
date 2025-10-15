from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


# Create specialized agents for different pipeline stages
data_collector1 = Agent(
    agent_name="Data_Collector_1",
    system_prompt="You are a data collection specialist focused on gathering comprehensive information from various sources. You excel at research and information gathering.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

data_collector2 = Agent(
    agent_name="Data_Collector_2",
    system_prompt="You are a data collection specialist focused on gathering comprehensive information from various sources. You excel at research and information gathering.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

data_processor1 = Agent(
    agent_name="Data_Processor_1",
    system_prompt="You are a data processing specialist focused on cleaning, organizing, and structuring data. You excel at data transformation and preparation.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

data_processor2 = Agent(
    agent_name="Data_Processor_2",
    system_prompt="You are a data processing specialist focused on cleaning, organizing, and structuring data. You excel at data transformation and preparation.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst1 = Agent(
    agent_name="Analyst_1",
    system_prompt="You are an analysis specialist focused on extracting insights and patterns from processed data. You excel at statistical analysis and pattern recognition.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst2 = Agent(
    agent_name="Analyst_2",
    system_prompt="You are an analysis specialist focused on extracting insights and patterns from processed data. You excel at statistical analysis and pattern recognition.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    system_prompt="You are a synthesis specialist focused on combining multiple analyses into coherent conclusions. You excel at integration and synthesis.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

pipeline_coordinator = Agent(
    agent_name="Pipeline_Coordinator",
    system_prompt="You are a pipeline coordinator that manages the flow of work between stages, ensures dependencies are met, and coordinates parallel and sequential processing.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def multi_stage_pipeline_algorithm(agents, task, **kwargs):
    """
    A multi-stage pipeline algorithm that combines parallel and sequential processing.
    """
    data_collectors = [
        agent
        for agent in agents
        if "Data_Collector" in agent.agent_name
    ]
    data_processors = [
        agent
        for agent in agents
        if "Data_Processor" in agent.agent_name
    ]
    analysts = [
        agent for agent in agents if "Analyst" in agent.agent_name
    ]
    synthesizer_agent = next(
        agent for agent in agents if "Synthesizer" in agent.agent_name
    )
    coordinator_agent = next(
        agent for agent in agents if "Coordinator" in agent.agent_name
    )

    # Pipeline configuration
    pipeline_stages = [
        {
            "name": "data_collection",
            "type": "parallel",
            "agents": data_collectors,
            "description": "Collect data from multiple sources in parallel",
        },
        {
            "name": "data_processing",
            "type": "parallel",
            "agents": data_processors,
            "description": "Process collected data in parallel",
            "depends_on": ["data_collection"],
        },
        {
            "name": "analysis",
            "type": "parallel",
            "agents": analysts,
            "description": "Analyze processed data in parallel",
            "depends_on": ["data_processing"],
        },
        {
            "name": "synthesis",
            "type": "sequential",
            "agents": [synthesizer_agent],
            "description": "Synthesize all analyses into final output",
            "depends_on": ["analysis"],
        },
    ]

    # Initialize pipeline state
    pipeline_results = {}
    stage_outputs = {}
    dependencies_met = {}

    coordinator_agent.run(
        f"Starting multi-stage pipeline for: {task}"
    )

    # Execute pipeline stages
    for stage in pipeline_stages:
        stage_name = stage["name"]
        stage_type = stage["type"]
        stage_agents = stage["agents"]
        dependencies = stage.get("depends_on", [])

        coordinator_agent.run(
            f"Starting stage: {stage_name} ({stage_type})"
        )

        # Check dependencies
        if dependencies:
            all_deps_met = all(
                dependencies_met.get(dep, False)
                for dep in dependencies
            )
            if not all_deps_met:
                coordinator_agent.run(
                    f"Dependencies not met for {stage_name}: {dependencies}"
                )
                continue

        if stage_type == "parallel":
            # Parallel execution
            parallel_results = []
            for i, agent in enumerate(stage_agents):
                if stage_name == "data_collection":
                    prompt = f"""
                    As a data collector, gather comprehensive information about: {task}
                    
                    Focus on:
                    - Primary sources and data
                    - Secondary research and analysis
                    - Relevant case studies and examples
                    - Industry trends and insights
                    - Technical specifications and requirements
                    
                    Provide detailed, well-organized data collection results.
                    """
                elif stage_name == "data_processing":
                    # Process data from previous stage
                    collected_data = stage_outputs.get(
                        "data_collection", []
                    )
                    prompt = f"""
                    As a data processor, clean and organize this collected data:
                    
                    Collected Data: {collected_data}
                    Task: {task}
                    
                    Focus on:
                    - Data cleaning and validation
                    - Organization and structuring
                    - Categorization and classification
                    - Quality assessment and filtering
                    - Format standardization
                    
                    Provide processed, structured data.
                    """
                elif stage_name == "analysis":
                    # Analyze processed data
                    processed_data = stage_outputs.get(
                        "data_processing", []
                    )
                    prompt = f"""
                    As an analyst, analyze this processed data:
                    
                    Processed Data: {processed_data}
                    Task: {task}
                    
                    Focus on:
                    - Pattern recognition and insights
                    - Statistical analysis and trends
                    - Comparative analysis
                    - Risk and opportunity assessment
                    - Predictive analysis and forecasting
                    
                    Provide detailed analytical insights.
                    """

                result = agent.run(prompt)
                parallel_results.append(
                    {"agent": agent.agent_name, "result": result}
                )

            stage_outputs[stage_name] = parallel_results
            pipeline_results[stage_name] = {
                "type": "parallel",
                "results": parallel_results,
                "status": "completed",
            }

        elif stage_type == "sequential":
            # Sequential execution
            if stage_name == "synthesis":
                # Synthesize all previous results
                analysis_results = stage_outputs.get("analysis", [])
                processed_data = stage_outputs.get(
                    "data_processing", []
                )
                collected_data = stage_outputs.get(
                    "data_collection", []
                )

                synthesis_prompt = f"""
                As a synthesizer, create a comprehensive final output by combining all pipeline results:
                
                Original Task: {task}
                
                Collected Data: {collected_data}
                Processed Data: {processed_data}
                Analysis Results: {analysis_results}
                
                Create a comprehensive synthesis that:
                1. Integrates all findings
                2. Provides clear conclusions
                3. Offers actionable recommendations
                4. Addresses the original task completely
                5. Highlights key insights and implications
                
                Provide a well-structured, comprehensive final output.
                """

                synthesis_result = synthesizer_agent.run(
                    synthesis_prompt
                )

                stage_outputs[stage_name] = [
                    {
                        "agent": synthesizer_agent.agent_name,
                        "result": synthesis_result,
                    }
                ]

                pipeline_results[stage_name] = {
                    "type": "sequential",
                    "results": [
                        {
                            "agent": synthesizer_agent.agent_name,
                            "result": synthesis_result,
                        }
                    ],
                    "status": "completed",
                }

        # Mark stage as completed
        dependencies_met[stage_name] = True

        # Quality check for each stage
        quality_prompt = f"""
        Assess the quality of the {stage_name} stage output:
        
        Stage: {stage_name}
        Output: {stage_outputs.get(stage_name, [])}
        Task: {task}
        
        Rate the quality (1-10) and provide feedback on:
        1. Completeness
        2. Accuracy
        3. Relevance
        4. Clarity
        5. Overall quality
        
        Should this stage be approved to proceed to dependent stages?
        """

        quality_assessment = coordinator_agent.run(quality_prompt)

        pipeline_results[stage_name][
            "quality_assessment"
        ] = quality_assessment

        coordinator_agent.run(f"Completed stage: {stage_name}")

    # Final pipeline summary
    summary_prompt = f"""
    Provide a comprehensive summary of the multi-stage pipeline execution:
    
    Task: {task}
    Pipeline Results: {pipeline_results}
    Total Stages: {len(pipeline_stages)}
    
    Summarize:
    1. Overall pipeline performance
    2. Quality of each stage
    3. Key findings and insights
    4. Effectiveness of parallel vs sequential processing
    5. Recommendations for improvement
    """

    pipeline_summary = coordinator_agent.run(summary_prompt)

    return {
        "task": task,
        "pipeline_stages": pipeline_stages,
        "pipeline_results": pipeline_results,
        "stage_outputs": stage_outputs,
        "dependencies_met": dependencies_met,
        "pipeline_summary": pipeline_summary,
        "total_stages": len(pipeline_stages),
        "algorithm_type": "multi_stage_pipeline",
    }


# Multi-Stage Pipeline Algorithm
social_alg = SocialAlgorithms(
    name="Multi-Stage-Pipeline-Algorithm",
    description="Multi-stage pipeline algorithm with parallel and sequential processing",
    agents=[
        data_collector1,
        data_collector2,
        data_processor1,
        data_processor2,
        analyst1,
        analyst2,
        synthesizer,
        pipeline_coordinator,
    ],
    social_algorithm=multi_stage_pipeline_algorithm,
    verbose=True,
    max_execution_time=1500,  # 25 minutes for complex pipeline
)

if __name__ == "__main__":
    result = social_alg.run(
        "Conduct a comprehensive market analysis for electric vehicle adoption in urban areas"
    )

    print("=== MULTI-STAGE PIPELINE ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(f"Total Stages: {result.final_outputs['total_stages']}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Communication Steps: {result.total_steps}")

    print("\n=== PIPELINE STAGES ===")
    for stage in result.final_outputs["pipeline_stages"]:
        print(f"Stage: {stage['name']} ({stage['type']})")
        print(f"  Description: {stage['description']}")
        if "depends_on" in stage:
            print(f"  Dependencies: {stage['depends_on']}")
        print()

    print("\n=== STAGE RESULTS ===")
    for stage_name, stage_data in result.final_outputs[
        "pipeline_results"
    ].items():
        print(f"Stage: {stage_name} ({stage_data['type']})")
        print(f"  Status: {stage_data['status']}")
        print(f"  Results: {len(stage_data['results'])} outputs")
        if "quality_assessment" in stage_data:
            print(
                f"  Quality: {stage_data['quality_assessment'][:100]}..."
            )
        print()

    print("\n=== FINAL SYNTHESIS ===")
    synthesis_stage = result.final_outputs["pipeline_results"].get(
        "synthesis", {}
    )
    if "results" in synthesis_stage and synthesis_stage["results"]:
        final_output = synthesis_stage["results"][0]["result"]
        print(final_output[:500] + "...")

    print("\n=== PIPELINE SUMMARY ===")
    print(result.final_outputs["pipeline_summary"][:500] + "...")
