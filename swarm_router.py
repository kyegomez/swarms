from swarms import Agent, SwarmRouter


agents = [
    Agent(
        agent_name="test_agent_1",
        agent_description="test_agent_1_description",
        system_prompt="test_agent_1_system_prompt",
        model_name="gpt-4o",
    ),
    Agent(
        agent_name="test_agent_2",
        agent_description="test_agent_2_description",
        system_prompt="test_agent_2_system_prompt",
        model_name="gpt-4o",
    ),
    Agent(
        agent_name="test_agent_3",
        agent_description="test_agent_3_description",
        system_prompt="test_agent_3_system_prompt",
        model_name="gpt-4o",
    ),
]

router = SwarmRouter(agents=agents, swarm_type="SequentialWorkflow")

print(router.run("How are you doing?"))
