
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

class TokenCache:
    def __init__(self, cache_duration_minutes=30):
        self.token_cache = defaultdict(lambda: {"token": None, "expires": datetime.now()})
        self.cache_duration = timedelta(minutes=cache_duration_minutes)

    def get_token(self, agent_name):
        cached_token = self.token_cache[agent_name]
        if cached_token["token"] and cached_token["expires"] > datetime.now():
            print(f"Using cached token for {agent_name}.")
            return cached_token["token"]
        return None  # Token has expired or does not exist

    def set_token(self, agent_name, token):
        self.token_cache[agent_name] = {
            "token": token,
            "expires": datetime.now() + self.cache_duration,
        }

class AdaptiveAgentFactory:
    def __init__(self, model, token_cache, reflection_steps=2):
        self.model = model
        self.token_cache = token_cache
        self.reflection_steps = reflection_steps

    def create_agent(self, agent_name, system_prompt, task, memory):
        cached_token = self.token_cache.get_token(agent_name)
        if cached_token:
            return cached_token

        # Create new agent instance with unique parameters
        new_agent = Agent(
            agent_name=agent_name,
            system_prompt=system_prompt,
            agent_description=f"Adaptive agent for {task}",
            llm=self.model,
            max_loops=3,
            autosave=True,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path=f"{agent_name.lower().replace(' ', '_')}.json",
            user_name="adaptive_user",
            retry_attempts=2,
            context_length=200000,
            long_term_memory=memory,
        )

        # Generate a token for the new agent and cache it
        token = f"{agent_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.token_cache.set_token(agent_name, token)
        print(f"Created new agent {agent_name} with token {token}.")
        return new_agent
