import os

from swarms import OpenAIChat, Agent, AgentRearrange


llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=150,
)

# Purpose = To detect email spam using three different agents
agent1 = Agent(
    agent_name="EmailPreprocessor",
    system_prompt="Clean and prepare the incoming email for further analysis. Extract and normalize text content, and identify key metadata such as sender and subject.",
    llm=llm,
    max_loops=1,
    output_type=str,
    # tools=[],
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
)

agent2 = Agent(
    agent_name="FeatureExtractor",
    system_prompt="Analyze the prepared email and extract relevant features that can help in spam detection, such as keywords, sender reputation, and email structure.",
    llm=llm,
    max_loops=1,
    output_type=str,
    # tools=[],
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
)

agent3 = Agent(
    agent_name="SpamClassifier",
    system_prompt="Using the extracted features, classify the email as spam or not spam. Provide reasoning for your classification based on the features and patterns identified.",
    llm=llm,
    max_loops=1,
    output_type=str,
    # tools=[],
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
)


swarm = AgentRearrange(
    flow=f"{agent1.agent_name} -> {agent2.agent_name} -> {agent3.agent_name}",
    agents=[agent1, agent2, agent3],
    logging_enabled=True,
    max_loops=1,
)


# Task
task = """
Clean and prepare the incoming email for further analysis. Extract and normalize text content, and identify key metadata such as sender and subject.

Subject: Re: Important Update - Account Verification Needed!

Dear Kye,

We hope this email finds you well. Our records indicate that your account has not been verified within the last 12 months. To continue using our services without interruption, please verify your account details by clicking the link below:

Verify Your Account Now

Please note that failure to verify your account within 48 hours will result in temporary suspension. We value your security and privacy; hence, this step is necessary to ensure your account's safety.

If you have any questions or need assistance, feel free to contact our support team at [Support Email] or visit our Help Center.

Thank you for your prompt attention to this matter.

"""


swarm.run(task)
