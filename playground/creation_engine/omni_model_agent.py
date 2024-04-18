from swarms import Agent, Anthropic, tool

# Model
llm = Anthropic(
    temperature=0.1,
)


# Tools
@tool
def text_to_video(task: str):
    """
    Converts a given text task into an animated video.

    Args:
        task (str): The text task to be converted into a video.

    Returns:
        str: The path to the exported GIF file.
    """
    import torch
    from diffusers import (
        AnimateDiffPipeline,
        MotionAdapter,
        EulerDiscreteScheduler,
    )
    from diffusers.utils import export_to_gif
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    dtype = torch.float16

    step = 4  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(
        load_file(hf_hub_download(repo, ckpt), device=device)
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        base, motion_adapter=adapter, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear",
    )

    output = pipe(
        prompt=task, guidance_scale=1.0, num_inference_steps=step
    )
    out = export_to_gif(output.frames[0], "animation.gif")
    return out


# Agent
agent = Agent(
    agent_name="Devin",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=[text_to_video],
)

# Run the agent
out = agent("Create a vide of a girl coding AI wearing hijab")
print(out)
