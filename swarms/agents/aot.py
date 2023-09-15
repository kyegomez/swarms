import logging
import os
import time

import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAI:
    def __init__(
            self, 
            api_key, 
            strategy="cot", 
            evaluation_strategy="value", 
            api_base="", 
            api_model="",
        ):
        if api_key == "" or api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", "")  # if not set, use the default base path of "https://api.openai.com/v1"
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')
            
        if api_model == "" or api_model is None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def run(
            self, 
            prompt, 
            max_tokens, 
            temperature, 
            k=1, 
            stop=None
        ):
        while True:
            try:
                if self.use_chat_api:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = openai.ChatCompletion.create(
                        model=self.api_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    response = openai.Completion.create(
                        engine=self.api_model,
                        prompt=prompt,
                        n=k,
                        max_tokens=max_tokens,
                        stop=stop,
                        temperature=temperature,
                    )
                with open("openai.logs", 'a') as log_file:
                    log_file.write("\n" + "-----------" + '\n' +"Prompt : "+ prompt+"\n")
                return response
            except openai.error.RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)

    def openai_choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text
    
    def generate_text(self, prompt, k):
        if self.use_chat_api:
            thoughts = []
            for _ in range(k):
                response = self.run(prompt, 400, 0.5, k)
                text = self.openai_choice2text_handler(response.choices[0])
                thoughts += [text]
                # print(f'thoughts: {thoughts}')
            return thoughts
            
        else:
            response = self.run(prompt, 300, 0.5, k)
            thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
            return thoughts

    def generate_thoughts(
            self, 
            state, 
            k, 
            initial_prompt, 
            rejected_solutions=None
        ):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("New state generating thought:", state, "\n\n")
        prompt = f"""
        Accomplish the task below by decomposing it as many very explicit subtasks as possible, be very explicit and thorough denoted by 
        a search process, highlighted by markers ‘1’,..., ‘3’ as “first operations” guiding subtree exploration for the OBJECTIVE, 
        focus on the third subtree exploration. Produce prospective search steps (e.g., the subtree exploration ‘5. 11 + 1’) 
        and evaluates potential subsequent steps to either progress
        towards a solution or retrace to another viable subtree then be very thorough 
        and think atomically then provide solutions for those subtasks, 
        then return the definitive end result and then summarize it


        ########## OBJECTIVE
        {initial_prompt}
        ###################
        """
        thoughts = self.generate_text(prompt, k)
        # print(f"Generated thoughts: {thoughts}")
        return thoughts

        
    def generate_solution(self, 
                          initial_prompt, 
                          state, 
                          rejected_solutions=None):
        try:
                
            if isinstance(state, list):
                state_text = '\n'.join(state)
            else:
                state_text = state
            
            prompt = f"""
            Generate a series of solutions to comply with the user's instructions, 
            you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, 
            while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, 
            Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            answer = self.generate_text(prompt, 1)
            print(f'Generated Solution Summary {answer}')
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                if (type(state) == str):
                    state_text = state
                else:
                    state_text = '\n'.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                response = self.run(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  
                state_values[state] = value
            return state_values

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

class AoTAgent:
    def __init__(
        self, 
        num_thoughts: int = None, 
        max_steps: int = None, 
        value_threshold: float = None, 
        pruning_threshold=0.5,
        backtracking_threshold=0.4,
        initial_prompt=None,
        openai_api_key: str = None,
        model = None,
    ):
        self.num_thoughts = num_thoughts
        self.max_steps = max_steps
        self.value_threshold = value_threshold
        self.backtracking_threshold = backtracking_threshold
        self.pruning_threshold = pruning_threshold
        self.initial_prompt = initial_prompt
        self.output = []
        self.openai_api_key = openai_api_key
        self.model = model
        self.model = self.model or OpenAI(api_key=self.openai_api_key)

    def solve(self):
        try:
            self.dfs(self.initial_prompt, 1)

            if not self.output:
                logger.error("No valid thoughts were generated during DFS")
                return None
            
            best_state, _ = max(self.output, key=lambda x: x[1])
            solution = self.model.generate_solution(self.initial_prompt, best_state)
            print(f"Solution is {solution}")
            return solution if solution else best_state
        except Exception as error:
            logger.error(f"Error in tot_dfs: {error}")
            raise error

    def dfs(self, state, step):
        if step > self.max_steps:
            thought, value = self.evaluate_thought(state)
            self.output.append((thought, value))
            return

        thoughts = self.generate_and_filter_thoughts(state)
        for next_state in thoughts:
            state_value = self.evaluated_thoughts[next_state]
            if state_value > self.value_threshold:
                child = (state, next_state) if isinstance(state, str) else (*state, next_state)
                self.dfs(child, step + 1)

                #backtracking
                best_value = max([value for _, value in self.output])
                if best_value < self.backtracking_threshold:
                    self.output.pop()
                    continue

    def generate_and_filter_thoughts(self, state):
        thoughts = self.model.generate_thoughts(
            state, 
            self.num_thoughts, 
            self.initial_prompt
        )

        self.evaluated_thoughts = self.model.evaluate_states(
            thoughts, 
            self.initial_prompt
        )

        filtered_thoughts = [thought for thought in thoughts if self.evaluated_thoughts[thought] >= self.pruning_threshold]
        print(f"filtered_thoughts: {filtered_thoughts}")
        return filtered_thoughts

    def evaluate_thought(self, state):
        thought = self.model.generate_thoughts(state, 1, self.initial_prompt)
        value = self.model.evaluate_states([state], self.initial_prompt)[state]
        print(f"Evaluated thought: {value}")
        return thought, value