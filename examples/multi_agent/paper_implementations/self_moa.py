from swarms import Agent

class SelfMoA:
    def __init__(self, query):
        # Model specification for classification
        self.model_spec = """
1. Multi-step logical problem solving and complex planning - claude opus 3
2. Mathematical problem solving and abstract reasoning - gpt-oss-120b
3. Code generation, refactoring, and debugging - gpt-5-codex
4. Creative writing/content generation and text summarization and question answering - claude-haiku-3
5. Conversational AI and virtual assistants - gpt-4.1
6. Research into specialized topics - o3-deep-research
7. Extraction of documents - claude-sonnet-3.5
8. Analysis of specialized information - gpt-5
9. Autonomous workflow orchestration/agentic tasks - claude-sonnet-4.5
"""
        self.model_choice = [
            "claude-3-opus-20240229",         # 1. Multi-step logical problem solving and complex planning
            "gpt-4.1",                        # 2. Mathematical problem solving and abstract reasoning
            "gpt-5-codex",                    # 3. Code generation, refactoring, and debugging
            "claude-3-haiku-20240307",        # 4. Creative writing/content generation and text summarization and question answering
            "gpt-4.1",                        # 5. Conversational AI and virtual assistants
            "gpt-5",                          # 6. Research into specialized topics
            "claude-3-5-sonnet-20240620",     # 7. Extraction of documents
            "gpt-5",                          # 8. Analysis of specialized information
            "claude-3-opus-20240229"          # 9. Autonomous workflow orchestration/agentic tasks
        ]
        self.query = query
        print("Developing Self-MoA Flow Response to '"+ query +"'")
        self.classifier_agent = Agent(
            agent_name="ProposerAgent",
            agent_description="Determines best model for the use-case/application",
            model_name="claude-3-opus-20240229",
            system_prompt=f"""You are a model classification agent. Your task is to analyze a user query and classify it into one of the following categories based on the provided model_spec.

{self.model_spec}

You MUST output ONLY the number corresponding to the best-fit model. Output a SINGLE INTEGER 1-9, nothing else: NO ADDITIONAL TEXT, NO EXPLANATION, NO PUNCTUATION.
""",
            max_loops=1,
            verbose=True,
        )
        self.selected_model_name = self._propose_model()
        self.main_agent_outputs = self._run_main_agents()
        self.final_answer = self._aggregate_outputs()

    def _propose_model(self):
        classification_result = self.classifier_agent.run(self.query)
        model_index = int(classification_result) - 1
        return self.model_choice[model_index]

    def _run_main_agents(self):
        main_agent_outputs = []
        num_runs = 6
        min_temp = 0
        temperature_step = 0.15
        for i in range(num_runs):
            temp = min_temp + i * temperature_step
            main_agent = Agent(
                agent_name="MainAgent",
                agent_description="Executes the task using the chosen model",
                model_name=self.selected_model_name,
                system_prompt="You are a helpful assistant. Please provide a detailed response to the user's query. Do it under 1500 characters.",
                max_loops=1,
                verbose=True,
                temperature=temp,
                max_tokens=2000,
            )
            print(f"\n\n\nRun {i+1} - Temperature: {temp:.2f} | Output saved.")
            response = main_agent.run(self.query)
            output_entry = {
                "temperature": temp,
                "response": response
            }
            main_agent_outputs.append(output_entry)
        return main_agent_outputs

    def _aggregate_outputs(self):
        aggregator_prompt = (
            "You are AggregatorAgent. You are given multiple outputs (potentially variations on detail, clarity, or order) produced by an assistant in response to the same query. "
            "Your task is to carefully examine all these outputs and synthesize them into a single, clear, best possible answer. "
            "Avoid unnecessary repetition, contradictions, and merge the best points from all responses.\n\n"
            "USER QUESTION:\n"
            f"{self.query}\n\n"
            "OUTPUTS:\n"
        )
        for entry in self.main_agent_outputs:
            aggregator_prompt += f"- (temperature={entry['temperature']:.2f}) {entry['response']}\n"
        aggregator_prompt += "\nFinal, best single answer:"
        aggregator_agent = Agent(
            agent_name="AggregatorAgent",
            agent_description="Aggregates and refines multiple agent outputs into a single, superior final answer.",
            model_name=self.selected_model_name,
            system_prompt=aggregator_prompt,
            max_loops=1,
            verbose=True,
        )
        print("\n\n\nRefined Aggregated Final Answer")
        final_answer = aggregator_agent.run("Please aggregate all outputs above into a clear, best possible answer to the user's question.")
        return final_answer

    def get_final_answer(self):
        return self.final_answer

# Example usage:
query = "Help me understand everything about Quantum Computing algorithms like the Shor's Algorithm. I need to do a school project on it."
SelfMoA(query)
