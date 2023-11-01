from swarms.models import OpenAIChat  # Replace with your actual OpenAIChat import

if __name__ == "__main__":
    api_key = ""  # Your OpenAI API key here
    agent = MultiTempAgent(api_key)

    prompt = "Write a blog post about health and wellness"
    final_output = agent.run(prompt)

    print("Final chosen output:")
    print(final_output)


class MultiTempAgent:
    def __init__(self, api_key, default_temp=0.5, alt_temps=[0.2, 0.7, 0.9]):
        self.api_key = api_key
        self.default_temp = default_temp
        self.alt_temps = alt_temps

    def ask_user_feedback(self, text):
        print(f"Generated text: {text}")
        feedback = input("Are you satisfied with this output? (yes/no): ")
        return feedback.lower() == "yes"

    def present_options_to_user(self, outputs):
        print("Alternative outputs:")
        for temp, output in outputs.items():
            print(f"Temperature {temp}: {output}")
        chosen_temp = float(input("Choose the temperature of the output you like: "))
        return outputs.get(chosen_temp, "Invalid temperature chosen.")

    def run(self, prompt):
        try:
            llm = OpenAIChat(openai_api_key=self.api_key, temperature=self.default_temp)
            initial_output = llm(prompt)  # Using llm as a callable
        except Exception as e:
            print(f"Error generating initial output: {e}")
            initial_output = None

        user_satisfied = self.ask_user_feedback(initial_output)

        if user_satisfied:
            return initial_output
        else:
            outputs = {}
            for temp in self.alt_temps:
                try:
                    llm = OpenAIChat(
                        openai_api_key=self.api_key, temperature=temp
                    )  # Re-initializing
                    outputs[temp] = llm(prompt)  # Using llm as a callable
                except Exception as e:
                    print(f"Error generating text at temperature {temp}: {e}")
                    outputs[temp] = None
            chosen_output = self.present_options_to_user(outputs)
            return chosen_output
