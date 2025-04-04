from swarms.utils.vllm_wrapper import VLLMWrapper


def main():
    # Initialize the vLLM wrapper with a model
    # Note: You'll need to have the model downloaded or specify a HuggingFace model ID
    llm = VLLMWrapper(
        model_name="meta-llama/Llama-2-7b-chat-hf",  # Replace with your model path or HF model ID
        temperature=0.7,
        max_tokens=1000,
    )

    # Example task
    task = "What are the benefits of using vLLM for inference?"

    # Run inference
    response = llm.run(task)
    print("Response:", response)

    # Example with system prompt
    llm_with_system = VLLMWrapper(
        model_name="meta-llama/Llama-2-7b-chat-hf",  # Replace with your model path or HF model ID
        system_prompt="You are a helpful AI assistant that provides concise answers.",
        temperature=0.7,
    )

    # Run inference with system prompt
    response = llm_with_system.run(task)
    print("\nResponse with system prompt:", response)

    # Example with batched inference
    tasks = [
        "What is vLLM?",
        "How does vLLM improve inference speed?",
        "What are the main features of vLLM?",
    ]

    responses = llm.batched_run(tasks, batch_size=2)
    print("\nBatched responses:")
    for task, response in zip(tasks, responses):
        print(f"\nTask: {task}")
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
