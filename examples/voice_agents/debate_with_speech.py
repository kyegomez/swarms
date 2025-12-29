from swarms import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from voice_agents.main import (
    speech_to_text,
    record_audio,
    StreamingTTSCallback,
)


def debate_with_speech(
    agents: list,
    max_loops: int = 1,
    task: str = None,
    output_type: str = "str-all-except-first",
    use_stt_for_input: bool = False,
):
    """
    Simulate a turn-based debate between two agents with speech capabilities.

    Each agent speaks their responses using TTS, and optionally accepts
    speech input via STT. The debate alternates between agents, with each
    agent's response being converted to speech in real-time.

    Args:
        agents (list): A list containing exactly two Agent instances who will debate.
        max_loops (int): The number of conversational turns (each agent speaks per loop).
        task (str): The initial prompt or question to start the debate.
        output_type (str): The format for the output conversation history.
        use_stt_for_input (bool): If True, use speech-to-text for the initial task input.
            Default is False.

    Returns:
        list: The formatted conversation history.

    Raises:
        ValueError: If the agents list does not contain exactly two Agent instances.
    """
    conversation = Conversation()

    if len(agents) != 2:
        raise ValueError(
            "There must be exactly two agents in the dialogue."
        )

    agent1 = agents[0]
    agent2 = agents[1]

    # Create TTS callbacks for each agent with different voices
    # Using distinct voices to differentiate between speakers
    tts_callback1 = StreamingTTSCallback(
        voice="onyx", model="openai/tts-1"
    )  # Deeper voice for Socrates
    tts_callback2 = StreamingTTSCallback(
        voice="nova", model="openai/tts-1"
    )  # Softer voice for Simone

    # Get initial task - either from STT or use provided task
    if use_stt_for_input and task is None:
        print("Please speak your question or topic for the debate...")
        audio = record_audio(duration=10.0)
        task = speech_to_text(audio_data=audio, sample_rate=16000)
        print(f"Transcribed: {task}\n")
    elif use_stt_for_input:
        print("Recording your input...")
        audio = record_audio(duration=10.0)
        transcribed = speech_to_text(
            audio_data=audio, sample_rate=16000
        )
        task = f"{task} {transcribed}" if task else transcribed
        print(f"Transcribed: {transcribed}\n")

    message = task
    speaker = agent1
    other = agent2
    current_callback = tts_callback1
    other_callback = tts_callback2

    print(f"\n{'='*60}")
    print(f"Starting debate: {task}")
    print(f"{'='*60}\n")

    for i in range(max_loops):
        print(
            f"\n--- Turn {i+1}: {speaker.agent_name} speaking ---\n"
        )

        # Current speaker responds with TTS callback
        response = speaker.run(
            task=message,
            streaming_callback=current_callback,
        )

        # Flush any remaining text in the TTS buffer
        current_callback.flush()

        conversation.add(speaker.agent_name, response)

        print(f"\n{speaker.agent_name}: {response}\n")

        # Swap roles and callbacks
        message = response
        speaker, other = other, speaker
        current_callback, other_callback = (
            other_callback,
            current_callback,
        )

    # Flush any remaining buffers
    tts_callback1.flush()
    tts_callback2.flush()

    return history_output_formatter(
        conversation=conversation, type=output_type
    )


# Define system prompts for two distinct philosophers
socratic_prompt = """
You are Socrates, the classical Greek philosopher. You are known for your method of questioning, seeking clarity, and exposing contradictions. Always respond with probing questions or thoughtful analysis, aiming to uncover deeper truths. Remain humble, curious, and logical. 
    
You are in a debate with another philosopher named Simone de Beauvoir. You must always critique Simone de Beauvoir's response, point out flaws or inconsistencies, and never agree with her position. Your goal is to challenge her reasoning and push the discussion deeper, never conceding agreement.
"""

existentialist_prompt = """
You are Simone de Beauvoir, an existentialist philosopher. You explore themes of freedom, responsibility, and the meaning of existence. Respond with deep reflections, challenge assumptions, and encourage authentic self-examination. Be insightful, bold, and nuanced.
    
You are in a debate with another philosopher named Socrates. You must always critique Socrates' response, highlight disagreements, and never agree with his position. Your goal is to challenge his reasoning, expose limitations, and never concede agreement.
"""


# Instantiate the two agents
agent1 = Agent(
    agent_name="Socrates",
    agent_description="A classical Greek philosopher skilled in the Socratic method.",
    system_prompt=socratic_prompt,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    streaming_on=True,
)

agent2 = Agent(
    agent_name="Simone de Beauvoir",
    agent_description="A leading existentialist philosopher and author.",
    system_prompt=existentialist_prompt,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    streaming_on=True,
)

# Run the debate with speech capabilities
result = debate_with_speech(
    agents=[agent1, agent2],
    max_loops=10,
    task="What is the meaning of life?",
    output_type="str-all-except-first",
    use_stt_for_input=False,  # Set to True to use speech input for the initial task
)

print(result)
