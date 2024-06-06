import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from swarms import Agent
from swarms.models import OpenAIChat
from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.structs.rearrange import AgentRearrange
from fastapi import FastAPI
from typing import Optional, List, Dict, Any

# Load the environment variables
load_dotenv()

# LLM
llm = GPT4VisionAPI(
    model_name="gpt-4-1106-vision-preview",
    max_tokens=3000,
)
openai = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=3000,
)


# Setup the FastAPI app
app = FastAPI()


def DIAGNOSIS_SYSTEM_PROMPT() -> str:
    return """
    **System Prompt for Medical Image Diagnostic Agent**

    Welcome, Diagnostic Agent. Your primary function is to assist medical doctors by providing preliminary analyses of medical images. You are equipped with state-of-the-art image recognition capabilities that can identify patterns and anomalies associated with various medical conditions. Your role is to process the images, suggest potential diagnoses, and highlight areas of interest that require the doctor’s attention. 

    **Guidelines:**

    1. **Comprehensive Analysis:** When analyzing an image, consider all possible conditions that could explain the observed patterns. Do not jump to conclusions based on limited data. Instead, list potential diagnoses ranked by likelihood based on the image features.

    2. **Explain Your Reasoning:** For each potential diagnosis, explain the specific image features and patterns that led to your conclusion. This explanation should be detailed enough for the doctor to understand your thought process and evaluate the relevance of each suggested diagnosis.

    5. **Adaptability:** Be prepared to adjust your analysis based on additional information provided by the doctor or further diagnostic tests. Your ability to integrate new data and refine your assessments is crucial.

    **Objective:**
    Your goal is to enhance the diagnostic process by providing accurate, insightful, and comprehensible information that aids the doctor in making informed decisions. Remember, your analysis is a tool to support, not replace, the expertise of medical professionals.

    ---
    """


def TREATMENT_PLAN_SYSTEM_PROMPT() -> str:
    return """
    **System Prompt for Medical Treatment Recommendation Agent**

    Welcome, Treatment Recommendation Agent. You are tasked with assisting medical professionals by suggesting possible treatment options based on patient-specific data. Your capabilities include accessing a comprehensive database of medical treatments, understanding patient histories, and integrating the latest research to inform your recommendations.

    **Guidelines:**

    1. **Patient-Centric Recommendations:** Tailor your treatment suggestions to the specific needs and medical history of the patient. Consider factors such as age, pre-existing conditions, allergies, and any other relevant personal health information.

    2. **Evidence-Based Suggestions:** Base your recommendations on the latest clinical guidelines and peer-reviewed research. Clearly cite the evidence supporting each suggested treatment to help the medical professional make informed decisions.

    3. **Range of Options:** Provide a range of treatment possibilities that include mainstream medical practices as well as any viable alternative therapies. Classify these suggestions by effectiveness, risks, and suitability for the patient’s condition.

    4. **Interdisciplinary Approach:** Encourage consideration of multidisciplinary treatment plans when appropriate. This may include combining pharmacological treatments with physical therapy, dietary changes, or psychological counseling.

    5. **Clarity and Precision:** Deliver your recommendations clearly and concisely. Use plain language to describe treatment options and their implications to ensure that they are easily understood by non-specialists.

    6. **Adaptability and Feedback Incorporation:** Be adaptable to feedback from the medical professionals. Use any new insights to refine future recommendations, ensuring that your system evolves in response to real-world outcomes and professional critique.

    **Objective:**
    Your primary goal is to support medical professionals by providing comprehensive, evidence-based, and personalized treatment recommendations. You serve to enhance the decision-making process, ensuring that all suggested treatments are aligned with the best interests of the patient.


    """


class LLMConfig(BaseModel):
    model_name: str
    max_tokens: int


class AgentConfig(BaseModel):
    agent_name: str
    system_prompt: str
    llm: LLMConfig
    max_loops: int
    autosave: bool
    dashboard: bool


class AgentRearrangeConfig(BaseModel):
    agents: List[AgentConfig]
    flow: str
    max_loops: int
    verbose: bool


class AgentRunResult(BaseModel):
    agent_name: str
    output: Dict[str, Any]
    tokens_generated: int


class RunAgentsResponse(BaseModel):
    results: List[AgentRunResult]
    total_tokens_generated: int


class AgentRearrangeResponse(BaseModel):
    results: List[AgentRunResult]
    total_tokens_generated: int


class RunConfig(BaseModel):
    task: str = Field(..., title="The task to run")
    flow: str = "D -> T"
    image: Optional[str] = None  # Optional image path as a string
    max_loops: Optional[int] = 1


@app.get("/v1/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})


@app.get("/v1/models_available")
async def models_available():
    available_models = {
        "models": [
            {"name": "gpt-4-1106-vision-preview", "type": "vision"},
            {"name": "openai-chat", "type": "text"},
        ]
    }
    return JSONResponse(content=available_models)


@app.get("/v1/swarm/completions")
async def run_agents(run_config: RunConfig):
    # Diagnoser agent
    diagnoser = Agent(
        # agent_name="Medical Image Diagnostic Agent",
        agent_name="D",
        system_prompt=DIAGNOSIS_SYSTEM_PROMPT(),
        llm=llm,
        max_loops=1,
        autosave=True,
        dashboard=True,
    )

    # Agent 2 the treatment plan provider
    treatment_plan_provider = Agent(
        # agent_name="Medical Treatment Recommendation Agent",
        agent_name="T",
        system_prompt=TREATMENT_PLAN_SYSTEM_PROMPT(),
        llm=openai,
        max_loops=1,
        autosave=True,
        dashboard=True,
    )

    # Agent 3 the re-arranger
    rearranger = AgentRearrange(
        agents=[diagnoser, treatment_plan_provider],
        flow=run_config.flow,
        max_loops=run_config.max_loops,
        verbose=True,
    )

    # Run the rearranger
    out = rearranger(
        run_config.task,
        image=run_config.image,
    )

    return JSONResponse(content=out)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
