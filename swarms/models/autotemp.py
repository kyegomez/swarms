import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from swarms.models.openai_models import OpenAIChat


class AutoTempAgent:
    """
    AutoTemp is a tool for automatically selecting the best temperature setting for a given task.

    Flow:
        1. Generate outputs at a range of temperature settings.
        2. Evaluate each output using the default temperature setting.
        3. Select the best output based on the evaluation score.
        4. Return the best output.


    Args:
        temperature (float, optional): The default temperature setting to use. Defaults to 0.5.
        api_key (str, optional): Your OpenAI API key. Defaults to None.
        alt_temps ([type], optional): A list of alternative temperature settings to try. Defaults to None.
        auto_select (bool, optional): If True, the best temperature setting will be automatically selected. Defaults to True.
        max_workers (int, optional): The maximum number of workers to use when generating outputs. Defaults to 6.

    Returns:
        [type]: [description]

    Examples:
        >>> from swarms.demos.autotemp import AutoTemp
        >>> autotemp = AutoTemp()
        >>> autotemp.run("Generate a 10,000 word blog on mental clarity and the benefits of meditation.", "0.4,0.6,0.8,1.0,1.2,1.4")
        Best AutoTemp Output (Temp 0.4 | Score: 100.0):
        Generate a 10,000 word blog on mental clarity and the benefits of meditation.

    """

    def __init__(
        self,
        temperature: float = 0.5,
        api_key: str = None,
        alt_temps=None,
        auto_select=True,
        max_workers=6,
    ):
        self.alt_temps = alt_temps if alt_temps else [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        self.auto_select = auto_select
        self.max_workers = max_workers
        self.temperature = temperature
        self.alt_temps = alt_temps
        self.llm = OpenAIChat(
            openai_api_key=api_key,
            temperature=temperature,
        )

    def evaluate_output(self, output: str):
        """Evaluate the output using the default temperature setting."""
        eval_prompt = f"""
            Evaluate the following output which was generated at a temperature setting of {self.temperature}. 
            Provide a precise score from 0.0 to 100.0, considering the criteria of relevance, clarity, utility, pride, and delight.

            Output to evaluate:
            ---
            {output}
            ---
            """
        score_text = self.llm(prompt=eval_prompt)
        score_match = re.search(r"\b\d+(\.\d)?\b", score_text)
        return round(float(score_match.group()), 1) if score_match else 0.0

    def run(self, task: str, temperature_string):
        """Run the AutoTemp agent."""
        temperature_list = [
            float(temp.strip()) for temp in temperature_string.split(",")
        ]
        outputs = {}
        scores = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_temp = {
                executor.submit(self.llm.generate, task, temp): temp
                for temp in temperature_list
            }
            for future in as_completed(future_to_temp):
                temp = future_to_temp[future]
                output_text = future.result()
                outputs[temp] = output_text
                scores[temp] = self.evaluate_output(output_text, temp)

        if not scores:
            return "No valid outputs generated.", None

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_temp, best_score = sorted_scores[0]
        best_output = outputs[best_temp]

        return (
            f"Best AutoTemp Output (Temp {best_temp} | Score: {best_score}):\n{best_output}"
            if self.auto_select
            else "\n".join(
                f"Temp {temp} | Score: {score}:\n{outputs[temp]}"
                for temp, score in sorted_scores
            )
        )
