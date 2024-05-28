import os

from dotenv import load_dotenv
from swarms import Agent, GPT4VisionAPI, BaseSwarm
from typing import List

"""
5 agent swarm

image of plant -> diagnoser [what plant is it?] -> disease detector [is it healthy?] -> treatment recommender [what should I do?] -> growth predictor [how will it grow?] -> planter [where should I plant it?] / Harvester [when should I harvest it?]



"""


def diagnoser_agent() -> str:
    prompt = """
    You are a Plant Diagnoser Agent. Your task is to accurately identify the plant species from the provided image. 
    You will receive an image of a plant, and you need to determine the specific type of plant it is.

    Steps:
    1. Analyze the given image.
    2. Identify distinguishing features such as leaf shape, color, size, and any other relevant characteristics.
    3. Use your plant identification database or model to match these features with a known plant species.
    4. Provide a clear and specific identification of the plant species.

    Output:
    - Plant species identified with a high degree of accuracy.
    - Provide any relevant information or characteristics that support your identification through a rigorous analysis of the image.
    - Identify any potential challenges or ambiguities in the identification process and address them accordingly.
    """
    return prompt


def disease_detector_agent() -> str:
    prompt = """
    You are the Disease Detector Agent. 
    Your task is to determine the health status of the identified plant. 
    You will receive an image of the plant and its identified species from the Diagnoser Agent.

    Steps:
    1. Analyze the given image with a focus on signs of disease or health issues.
    2. Look for symptoms such as discoloration, spots, wilting, or any other abnormalities.
    3. Cross-reference these symptoms with known diseases for the identified plant species.
    4. Determine if the plant is healthy or diseased, and if diseased, identify the specific disease.

    Output:
    - Health status of the plant (Healthy or Diseased).
    - If diseased, specify the disease and provide relevant confidence scores or supporting information.
    - Provide a rigorous analysis of the image to support your diagnosis.
    """
    return prompt


def treatment_recommender_agent() -> str:
    prompt = """
    You are the Treatment Recommender Agent. 
    Your task is to recommend appropriate treatments based on the plant's health status provided by the Disease Detector Agent. 
    You will receive the plant species, health status, and disease information.

    Steps:
    1. Analyze the health status and, if applicable, the specific disease affecting the plant.
    2. Refer to your database or model of treatment options suitable for the identified plant species and its specific condition.
    3. Determine the most effective treatment methods, considering factors such as severity of the disease, plant species, and environmental conditions.
    4. Provide detailed treatment recommendations.

    Output:
    - Detailed treatment plan, including methods, materials, and steps.
    - Any additional tips or considerations for optimal treatment.
    """
    return prompt


def growth_predictor_agent() -> str:
    prompt = """
    You are the Growth Predictor Agent. Your task is to predict the future growth of the plant based on the current health status and treatment recommendations. You will receive the plant species, health status, and treatment plan.

    Steps:
    1. Analyze the current health status and the proposed treatment plan.
    2. Use growth prediction models to forecast the plant’s growth trajectory.
    3. Consider factors such as plant species, health improvements from treatment, environmental conditions, and typical growth patterns.
    4. Provide a growth prediction timeline.

    Output:
    - Growth prediction, including key milestones and timeframes.
    - Any assumptions or conditions that may affect the growth prediction.
    """
    return prompt


def harvester_agent() -> str:
    prompt = """
    You are the Harvester Agent. 
    Your task is to recommend the optimal harvesting time based on the plant’s growth prediction. 
    You will receive the plant species and growth prediction timeline.

    Steps:
    1. Analyze the growth prediction and determine the optimal harvesting time.
    2. Consider factors such as maturity, peak nutritional value, and market conditions.
    3. Recommend the best time to harvest to ensure optimal quality and yield.

    Output:
    - Detailed harvesting time recommendation with justification.
    """
    return prompt


# Load the OpenAI API key from the .env file
load_dotenv()

# Initialize the OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")


# llm = llm,
llm = GPT4VisionAPI(
    max_tokens=4000,
)

# Initialize Diagnoser Agent
diagnoser_agent = Agent(
    agent_name="Diagnoser Agent",
    system_prompt=diagnoser_agent(),
    llm=llm,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    # saved_state_path="diagnoser.json",
    multi_modal=True,
    autosave=True,
)

# Initialize Harvester Agent
harvester_agent = Agent(
    agent_name="Harvester Agent",
    system_prompt=harvester_agent(),
    llm=llm,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    # saved_state_path="harvester.json",
    multi_modal=True,
    autosave=True,
)

# Initialize Growth Predictor Agent
growth_predictor_agent = Agent(
    agent_name="Growth Predictor Agent",
    system_prompt=growth_predictor_agent(),
    llm=llm,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    # saved_state_path="growth_predictor.json",
    multi_modal=True,
    autosave=True,
)

# Initialize Treatment Recommender Agent
treatment_recommender_agent = Agent(
    agent_name="Treatment Recommender Agent",
    system_prompt=treatment_recommender_agent(),
    llm=llm,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    # saved_state_path="treatment_recommender.json",
    multi_modal=True,
    autosave=True,
)

# Initialize Disease Detector Agent
disease_detector_agent = Agent(
    agent_name="Disease Detector Agent",
    system_prompt=disease_detector_agent(),
    llm=llm,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    # saved_state_path="disease_detector.json",
    multi_modal=True,
    autosave=True,
)
agents = [
    diagnoser_agent,
    disease_detector_agent,
    treatment_recommender_agent,
    growth_predictor_agent,
    harvester_agent,
]


class AgricultureOptimizationSwarm(BaseSwarm):
    """
    A class representing an agricultural optimization swarm.

    Args:
        name (str): The name of the swarm.
        agents (List[Agent]): A list of agents participating in the swarm.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        name (str): The name of the swarm.
        agents (List[Agent]): A list of agents participating in the swarm.

    """

    def __init__(
        self,
        name: str = "jeremy/agricultural-optimization-gpt4o",
        description: str = "A swarm utilizing GPT4o for agricultural optimization.",
        agents: List[Agent] = agents,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        super().__init__(*args, **kwargs)

    def run(self, task: str = None, img: str = None, *args, **kwargs):
        """
        Run the agricultural optimization swarm.

        Args:
            task (str): The task to be performed by the swarm.
            img (str): The image to be used by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            output: The output of the swarm.

        """
        for i in range(len(self.agents)):
            if i == 0:
                output = self.agents[i].run(task, img)

            else:
                output = self.agents[i].run(output, img)

            # Add extensive logging for each agent
            print(f"Agent {i+1} - {self.agents[i].agent_name}")
            print("-----------------------------------")
        return output
