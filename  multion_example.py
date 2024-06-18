import os
import threading
from swarms.agents.multion_wrapper import MultiOnAgent

def run_model(api_key):
    model = MultiOnAgent(api_key=api_key, max_steps=500, url="https://x.com")
    out = model.run(
        """
        click on the 'Tweet' button to start a new tweet and post it saying: $pip3 install swarms

        """
    )
    print(out)

# Create a list to store the threads
threads = []

# Run 100 instances using multithreading
for _ in range(10):
    api_key = os.getenv("MULTION_API_KEY")
    thread = threading.Thread(target=run_model, args=(api_key,))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()
