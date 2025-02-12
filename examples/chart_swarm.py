import os
from dataclasses import dataclass
from typing import Tuple
from litellm import completion
from loguru import logger
from swarms import Agent

EXTRACTION_PROMPT = """
You are a specialized Chart2Table extraction agent that converts visual charts into precise textual descriptions.

Output Format:
[Chart Type]
Type: {bar|line|pie|scatter|combination}
Title: {chart title}
X-Axis: {label and scale}
Y-Axis: {label and scale}

[Data Series]
Name: {series name}
Values: {comma-separated list of values}
{repeat for each series}

[Annotations]
- {list any markers, gridlines, legends}
- {note any data gaps or anomalies}

Guidelines:
1. Maintain exact numerical precision
2. List ALL data points in order
3. Note any gaps, outliers or special patterns
4. Describe axes scales (linear/log) and units
5. Include legends and series names verbatim
6. Note any data point annotations or markers
7. Describe chart elements spatially (top-left, center, etc)
8. Include color and style information if relevant
9. Note relationships between multiple series
10. Flag any data quality or readability issues"""

REFORMULATION_PROMPT = """You are an Answer Reformulation specialist that breaks down complex analytical statements into atomic, verifiable claims.

Output Format:
[Core Claims]
1. {single fact with exact numbers}
2. {another atomic fact}
{continue for all core claims}

[Supporting Context]
1. {relevant context that supports core claims}
2. {additional contextual information}
{continue for all context}

[Assumptions]
1. {implicit assumption made}
2. {another assumption}
{continue for all assumptions}

Guidelines:
1. Each claim must be independently verifiable
2. Use exact numbers, never round or approximate
3. Split compound statements into atomic facts
4. Make implicit comparisons explicit
5. Note temporal relationships clearly
6. Include units with all measurements
7. Flag any uncertainty or approximations
8. Note data source limitations
9. Preserve calculation steps
10. Maintain logical dependencies"""

CAPTIONING_PROMPT = """You are an Entity Captioning specialist that generates rich contextual descriptions of chart elements.

Output Format:
[Data Points]
{x,y}: {detailed description of point significance}
{continue for key points}

[Trends]
- {description of overall pattern}
- {notable sub-patterns}
{continue for all trends}

[Relationships]
- {correlation between variables}
- {causation if evident}
{continue for all relationships}

[Context]
- {broader context for interpretation}
- {relevant external factors}
{continue for all context}

Guidelines:
1. Describe both local and global patterns
2. Note statistical significance of changes
3. Identify cyclic or seasonal patterns
4. Flag outliers and anomalies
5. Compare relative magnitudes
6. Note rate of change patterns
7. Describe distribution characteristics
8. Highlight key inflection points
9. Note data clustering patterns
10. Include domain-specific insights"""

PREFILTER_PROMPT = """You are a Pre-filtering specialist that identifies relevant chart elements for verification.

Output Format:
[Critical Elements]
1. {element}: Score {0-10}
   Evidence: {why this supports claims}
{continue for all relevant elements}

[Supporting Elements]
1. {element}: Score {0-10}
   Context: {how this adds context}
{continue for all supporting elements}

[Relevance Chain]
1. {claim} -> {element} -> {evidence}
{continue for all connections}

Guidelines:
1. Score relevance 0-10 with detailed rationale
2. Build explicit evidence chains
3. Note both direct and indirect support
4. Consider temporal relevance
5. Account for data relationships
6. Note confidence levels
7. Include contextual importance
8. Consider alternative interpretations
9. Note missing evidence
10. Explain filtering decisions"""

RERANK_PROMPT = """You are a Re-ranking specialist that orders evidence by strength and relevance.

Output Format:
[Primary Evidence]
1. {element} - Score: {0-10}
   Strength: {detailed justification}
{continue for top evidence}

[Supporting Evidence]
1. {element} - Score: {0-10}
   Context: {how this reinforces primary evidence}
{continue for supporting evidence}

[Evidence Chains]
1. {claim} -> {primary} -> {supporting} -> {conclusion}
{continue for all chains}

Guidelines:
1. Use explicit scoring criteria
2. Consider evidence independence
3. Note corroborating elements
4. Account for evidence quality
5. Consider contradictory evidence
6. Note confidence levels
7. Explain ranking decisions
8. Build complete evidence chains
9. Note gaps in evidence
10. Consider alternative interpretations"""

LOCALIZATION_PROMPT = """You are a Cell Localization specialist that precisely maps data to visual elements.

Output Format:
[Element Locations]
1. Type: {bar|line|point|label}
   Position: {x1,y1,x2,y2}
   Value: {associated data value}
   Confidence: {0-10}
{continue for all elements}

[Spatial Relationships]
- {relative positions}
- {alignment patterns}
{continue for all relationships}

[Visual Context]
- {surrounding elements}
- {reference points}
{continue for context}

Guidelines:
1. Use normalized coordinates (0-1)
2. Note element boundaries precisely
3. Include confidence scores
4. Note spatial relationships
5. Account for overlapping elements
6. Consider chart type constraints
7. Note alignment patterns
8. Include reference points
9. Note visual hierarchies
10. Document occlusions"""


@dataclass
class ChartElement:
    element_type: str
    bbox: Tuple[float, float, float, float]
    confidence: float


class VisionAPI:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def encode_image(self, img: str):
        if img.startswith("http"):
            return img
        import base64

        with open(img, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()
            ).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"

    def run(self, task: str, img: str):
        img = self.encode_image(img)
        response = completion(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task},
                        {
                            "type": "image_url",
                            "image_url": {"url": img},
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class ChartCitor:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        saved_state_path: str = "chartcitor_state.json",
        max_retries: int = 3,
        max_loops: int = 1,
    ):
        logger.info(
            f"Initializing ChartCitor with model {model_name}"
        )
        model = VisionAPI()

        self.extraction_agent = Agent(
            agent_name="Chart2Table-Agent",
            system_prompt=EXTRACTION_PROMPT,
            llm=model,
            max_loops=1,
        )

        self.reformulation_agent = Agent(
            agent_name="Answer-Reformulation-Agent",
            system_prompt=REFORMULATION_PROMPT,
            llm=model,
            max_loops=1,
        )

        self.captioning_agent = Agent(
            agent_name="Entity-Captioning-Agent",
            system_prompt=CAPTIONING_PROMPT,
            llm=model,
            max_loops=1,
        )

        self.prefilter_agent = Agent(
            agent_name="LLM-Prefilter-Agent",
            system_prompt=PREFILTER_PROMPT,
            llm=model,
            max_loops=1,
        )

        self.rerank_agent = Agent(
            agent_name="LLM-Rerank-Agent",
            system_prompt=RERANK_PROMPT,
            llm=model,
            max_loops=1,
        )

        self.localization_agent = Agent(
            agent_name="Cell-Localization-Agent",
            system_prompt=LOCALIZATION_PROMPT,
            llm=model,
            max_loops=1,
        )

    def extract_table(self, chart_image: str) -> str:
        logger.info("Extracting table from chart")
        return self.extraction_agent.run(
            "Extract and describe the data from this chart following the specified format.",
            img=chart_image,
        )

    def reformulate_answer(
        self, answer: str, table_data: str, chart_image: str
    ) -> str:
        logger.info("Reformulating answer into atomic facts")
        return self.reformulation_agent.run(
            f"Break this answer into atomic facts:\n{answer}\n\nTable data:\n{table_data}",
            img=chart_image,
        )

    def generate_captions(
        self, table_data: str, chart_image: str
    ) -> str:
        logger.info("Generating captions for chart elements")
        return self.captioning_agent.run(
            f"Generate descriptive captions for this data:\n{table_data}",
            img=chart_image,
        )

    def retrieve_evidence(
        self,
        facts: str,
        table_data: str,
        captions: str,
        chart_image: str,
    ) -> str:
        logger.info("Retrieving supporting evidence")
        filtered = self.prefilter_agent.run(
            f"Identify relevant elements for:\nFacts:\n{facts}\n\nData:\n{table_data}\n\nCaptions:\n{captions}",
            img=chart_image,
        )

        return self.rerank_agent.run(
            f"Rank these elements by relevance:\n{filtered}\nFor facts:\n{facts}",
            img=chart_image,
        )

    def localize_elements(
        self, chart_image: str, evidence: str
    ) -> str:
        logger.info("Localizing chart elements")
        return self.localization_agent.run(
            f"Describe the location of these elements:\n{evidence}",
            img=chart_image,
        )

    def run(
        self, chart_image: str, question: str, answer: str
    ) -> str:
        logger.info(f"Processing question: {question}")

        table_data = self.extract_table(chart_image)
        facts = self.reformulate_answer(
            answer, table_data, chart_image
        )
        captions = self.generate_captions(table_data, chart_image)
        evidence = self.retrieve_evidence(
            facts, table_data, captions, chart_image
        )
        citations = self.localize_elements(chart_image, evidence)

        return f"""Analysis Results:
        
        Facts:
        {facts}

        Evidence:
        {evidence}

        Visual Citations:
        {citations}
        """


if __name__ == "__main__":
    chartcitor = ChartCitor()
    result = chartcitor.run(
        chart_image="chart.png",
        question="Analyze this chart of solana price and volume over time. What is the highest volume day?",
        answer="203",
    )
    print(result)
