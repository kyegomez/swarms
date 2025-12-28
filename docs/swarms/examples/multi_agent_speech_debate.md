# Multi-Agent Speech Debate

This tutorial explores a more advanced use case: simulating a turn-based debate between two agents where each agent speaks their responses. We will also optionally use Speech-to-Text (STT) to provide the initial debate topic.

## Prerequisites

- Python 3.10+
- OpenAI API key
- `swarms` library
- `voice-agents` library
- A working microphone (if using STT)

## Tutorial Steps

1. **Install Dependencies**
   ```bash
   pip3 install -U swarms voice-agents
   ```

2. **Define Agent Personalities**
   Create distinct system prompts for your agents to ensure a dynamic debate. In this example, we use Socrates and Simone de Beauvoir.

3. **Initialize Agents**
   Set up two agents with `streaming_on=True`.

4. **Create a Debate Loop**
   Implement a function that alternates turns between agents, uses their respective TTS voices, and passes the response of one agent as the input to the next.

5. **Integrate STT (Optional)**
   Use `record_audio` and `speech_to_text` to capture your own voice as the starting prompt for the debate.

## Code Example

```python
from swarms import Agent
from swarms.structs.conversation import Conversation
from voice_agents.main import speech_to_text, record_audio, StreamingTTSCallback

def debate_with_speech(
    agents: list,
    max_loops: int = 1,
    task: str = None,
    use_stt_for_input: bool = False,
):
    """
    Simulate a turn-based debate between two agents with speech capabilities.
    
    Args:
        agents (list): A list containing exactly two Agent instances who will debate.
        max_loops (int): The number of conversational turns.
        task (str): The initial prompt or question to start the debate.
        use_stt_for_input (bool): If True, use speech-to-text for the initial task input.
    
    Returns:
        str: The formatted conversation history.
    """
    conversation = Conversation()
    
    # Create TTS callbacks with different voices to differentiate speakers
    tts_callback1 = StreamingTTSCallback(voice="onyx", model="tts-1")  # Deeper voice
    tts_callback2 = StreamingTTSCallback(voice="nova", model="tts-1")   # Softer voice
    
    # Get initial task from STT or provided string
    if use_stt_for_input:
        print("Please speak your question or topic for the debate...")
        audio = record_audio(duration=5.0)
        task = speech_to_text(audio_data=audio, sample_rate=16000)
        print(f"Transcribed: {task}\n")
    
    message = task
    speaker = agents[0]
    other = agents[1]
    current_callback = tts_callback1
    other_callback = tts_callback2
    
    for i in range(max_loops):
        print(f"--- Turn {i+1}: {speaker.agent_name} speaking ---")
        
        # Agent generates response and speaks in real-time
        response = speaker.run(
            task=message,
            streaming_callback=current_callback,
        )
        current_callback.flush()
        
        conversation.add(speaker.agent_name, response)
        
        # Swap roles for the next turn
        message = response
        speaker, other = other, speaker
        current_callback, other_callback = other_callback, current_callback
    
    return conversation.return_history_as_string()

# Define System Prompts
socratic_prompt = "You are Socrates. Challenge every assumption with logic."
beauvoir_prompt = "You are Simone de Beauvoir. Focus on freedom and existence."

# Instantiate Agents
agent1 = Agent(
    agent_name="Socrates",
    system_prompt=socratic_prompt,
    model_name="gpt-4o",
    streaming_on=True,
)
agent2 = Agent(
    agent_name="Simone de Beauvoir",
    system_prompt=beauvoir_prompt,
    model_name="gpt-4o",
    streaming_on=True,
)

# Run the debate
history = debate_with_speech(
    agents=[agent1, agent2],
    max_loops=3,
    task="Is freedom an illusion?",
)

print(history)
```

## Key Components

- **Differentiated Voices**: Using "onyx" and "nova" helps the listener distinguish which agent is currently speaking.
- **Turn-based Logic**: The output of the first agent becomes the input for the second, creating a continuous dialogue.
- **STT Integration**: `speech_to_text` allows for hands-free interaction with the swarm.
- **Conversation Tracking**: The `Conversation` struct helps maintain a record of the entire exchange.

