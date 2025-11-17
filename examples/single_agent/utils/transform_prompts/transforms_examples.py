"""
Message Transforms Examples for Swarms

This file demonstrates various ways to use the message transforms feature
for handling context size limitations and message count restrictions in LLMs.
"""

from swarms import Agent
from swarms.structs.transforms import (
    MessageTransforms,
    TransformConfig,
    TransformResult,
    create_default_transforms,
    apply_transforms_to_messages,
)

from loguru import logger


def example_1_basic_transforms():
    """Example 1: Basic transforms usage with TransformConfig"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Transforms with TransformConfig")
    print("=" * 60)

    # Create sample messages that exceed typical context limits
    sample_messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant specializing in software development.",
        },
        {
            "role": "user",
            "content": "Explain the concept of object-oriented programming.",
        },
        {
            "role": "assistant",
            "content": "Object-oriented programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data and code. The data is in the form of fields (often called attributes or properties), and the code is in the form of procedures (often called methods). OOP focuses on the objects that developers want to manipulate rather than the logic required to manipulate them.",
        },
        {
            "role": "user",
            "content": "Can you give me a simple example in Python?",
        },
        {
            "role": "assistant",
            "content": "Here's a simple example of a class in Python:\n\n```python\nclass Dog:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\n    def bark(self):\n        return f'{self.name} says woof!'\n\n# Usage\nmy_dog = Dog('Buddy', 3)\nprint(my_dog.bark())  # Output: Buddy says woof!\n```\n\nThis example shows:\n1. A `Dog` class with attributes `name` and `age`\n2. An `__init__` method (constructor)\n3. A `bark` method\n4. How to create an instance and call methods",
        },
        {
            "role": "user",
            "content": "What are the four main principles of OOP?",
        },
        {
            "role": "assistant",
            "content": "The four main principles of Object-Oriented Programming are:\n\n1. **Encapsulation**: Bundling data and methods that operate on that data within a single unit (class). It restricts direct access to some of an object's components.\n\n2. **Abstraction**: Hiding complex implementation details and showing only the necessary features of an object. This simplifies the interface.\n\n3. **Inheritance**: Creating new classes from existing classes, allowing code reuse and establishing a relationship between parent and child classes.\n\n4. **Polymorphism**: The ability of different objects to respond to the same method call in different ways. This allows for more flexible and extensible code.",
        },
        {
            "role": "user",
            "content": "How does inheritance work in practice?",
        },
        {
            "role": "assistant",
            "content": "Let me show you a practical example of inheritance:\n\n```python\nclass Animal:\n    def __init__(self, name):\n        self.name = name\n\n    def speak(self):\n        pass  # Abstract method\n\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} says Woof!'\n\nclass Cat(Animal):\n    def speak(self):\n        return f'{self.name} says Meow!'\n\n# Usage\ndog = Dog('Buddy')\ncat = Cat('Whiskers')\nprint(dog.speak())  # Buddy says Woof!\nprint(cat.speak())  # Whiskers says Meow!\n```\n\nHere, `Dog` and `Cat` inherit from `Animal`. They both have a `speak` method, but each implements it differently - this is polymorphism in action!",
        },
        {
            "role": "user",
            "content": "What are some real-world applications of OOP?",
        },
        {
            "role": "assistant",
            "content": "Object-oriented programming is used extensively in real-world applications:\n\n1. **GUI Applications**: Frameworks like Qt, Swing, and WPF use OOP to create user interfaces with windows, buttons, and other components as objects.\n\n2. **Game Development**: Game engines like Unity and Unreal Engine heavily use OOP. Characters, weapons, environments, and game logic are all implemented as objects.\n\n3. **Web Applications**: Frameworks like Django (Python), Ruby on Rails, and ASP.NET use OOP patterns for handling requests, managing databases, and organizing code.\n\n4. **Database Systems**: Object-Relational Mapping (ORM) systems like SQLAlchemy, Hibernate, and Entity Framework map database tables to objects.\n\n5. **Operating Systems**: Modern operating systems use OOP for managing processes, files, network connections, and hardware devices.\n\n6. **Mobile Applications**: iOS (Objective-C/Swift) and Android (Java/Kotlin) development heavily relies on OOP concepts.\n\n7. **Scientific Computing**: Libraries like NumPy and SciPy, while not purely OOP, use objects to represent complex mathematical structures.\n\n8. **Financial Systems**: Banking software, trading platforms, and financial modeling tools use OOP to represent accounts, transactions, and complex financial instruments.\n\nThe benefits of OOP in these applications include:\n- Code reusability\n- Easier maintenance\n- Better organization\n- Encapsulation of complex logic\n- Inheritance for building upon existing functionality",
        },
    ]

    # Create transforms configuration
    config = TransformConfig(
        enabled=True,
        method="middle-out",
        model_name="gpt-4",
        preserve_system_messages=True,
        preserve_recent_messages=3,
        max_tokens=4000,  # Force compression for demonstration
    )

    # Create transforms instance
    transforms = MessageTransforms(config)

    # Apply transforms
    result: TransformResult = transforms.transform_messages(
        sample_messages
    )

    print(f"Original messages: {result.original_message_count}")
    print(f"Compressed messages: {result.compressed_message_count}")
    print(f"Original tokens: {result.original_token_count}")
    print(f"Compressed tokens: {result.compressed_token_count}")
    print(".2f")

    if result.was_compressed:
        print("\nTransformed messages:")
        for i, msg in enumerate(result.messages, 1):
            print(
                f"{i}. {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}"
            )
    else:
        print("No compression was needed.")


def example_2_dictionary_config():
    """Example 2: Using dictionary configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Dictionary Configuration")
    print("=" * 60)

    # Create transforms using dictionary (alternative to TransformConfig)
    dict_config = {
        "enabled": True,
        "method": "middle-out",
        "model_name": "claude-3-sonnet",
        "preserve_system_messages": True,
        "preserve_recent_messages": 2,
        "max_messages": 5,  # Force message count compression
    }

    config = TransformConfig(**dict_config)
    transforms = MessageTransforms(config)

    # Sample messages
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {
            "role": "user",
            "content": "Help me debug this Python code.",
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help! Please share your Python code and describe the issue you're experiencing.",
        },
        {
            "role": "user",
            "content": "Here's my code: def factorial(n): if n == 0: return 1 else: return n * factorial(n-1)",
        },
        {
            "role": "assistant",
            "content": "Your factorial function looks correct! It's a classic recursive implementation. However, it doesn't handle negative numbers. Let me suggest an improved version...",
        },
        {
            "role": "user",
            "content": "It works for positive numbers but crashes for large n due to recursion depth.",
        },
        {
            "role": "assistant",
            "content": "Ah, that's a common issue with recursive factorial functions. Python has a default recursion limit of 1000. For large numbers, you should use an iterative approach instead...",
        },
        {
            "role": "user",
            "content": "Can you show me the iterative version?",
        },
    ]

    result = transforms.transform_messages(messages)

    print(f"Original messages: {result.original_message_count}")
    print(f"Compressed messages: {result.compressed_message_count}")
    print(f"Compression applied: {result.was_compressed}")

    if result.was_compressed:
        print("\nCompressed conversation:")
        for msg in result.messages:
            print(
                f"{msg['role'].title()}: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}"
            )


def example_3_agent_integration():
    """Example 3: Integration with Agent class"""
    print("\n" + "=" * 60)
    print("Example 3: Agent Integration")
    print("=" * 60)

    # Create agent with transforms enabled
    agent = Agent(
        agent_name="Transformed-Agent",
        agent_description="AI assistant with automatic context management",
        model_name="gpt-4.1",
        max_loops=1,
        streaming_on=False,
        print_on=False,
        # Enable transforms
        transforms=TransformConfig(
            enabled=True,
            method="middle-out",
            model_name="gpt-4.1",
            preserve_system_messages=True,
            preserve_recent_messages=3,
        ),
    )

    print("Agent created with transforms enabled.")
    print(
        "The agent will automatically apply message transforms when context limits are approached."
    )

    # You can also check if transforms are active
    if agent.transforms is not None:
        print("✓ Transforms are active on this agent")
        print(f"  Method: {agent.transforms.config.method}")
        print(f"  Model: {agent.transforms.config.model_name}")
        print(
            f"  Preserve recent: {agent.transforms.config.preserve_recent_messages}"
        )
    else:
        print("✗ No transforms configured")


def example_4_convenience_function():
    """Example 4: Using convenience functions"""
    print("\n" + "=" * 60)
    print("Example 4: Convenience Functions")
    print("=" * 60)

    # Sample messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Tell me about machine learning.",
        },
        {
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions.",
        },
        {"role": "user", "content": "What are the main types?"},
        {
            "role": "assistant",
            "content": "There are three main types of machine learning:\n\n1. **Supervised Learning**: The algorithm learns from labeled training data. Examples include classification and regression tasks.\n\n2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data. Examples include clustering and dimensionality reduction.\n\n3. **Reinforcement Learning**: The algorithm learns through trial and error by interacting with an environment. Examples include game playing and robotic control.",
        },
        {"role": "user", "content": "Can you give examples of each?"},
    ]

    # Method 1: Using create_default_transforms
    print("Method 1: create_default_transforms")
    transforms = create_default_transforms(
        enabled=True,
        model_name="gpt-3.5-turbo",
    )
    result1 = transforms.transform_messages(messages)
    print(
        f"Default transforms - Original: {result1.original_message_count}, Compressed: {result1.compressed_message_count}"
    )

    # Method 2: Using apply_transforms_to_messages directly
    print("\nMethod 2: apply_transforms_to_messages")
    config = TransformConfig(
        enabled=True, max_tokens=1000
    )  # Force compression
    result2 = apply_transforms_to_messages(messages, config, "gpt-4")
    print(
        f"Direct function - Original tokens: {result2.original_token_count}, Compressed tokens: {result2.compressed_token_count}"
    )


def example_5_advanced_scenarios():
    """Example 5: Advanced compression scenarios"""
    print("\n" + "=" * 60)
    print("Example 5: Advanced Scenarios")
    print("=" * 60)

    # Scenario 1: Very long conversation with many messages
    print("Scenario 1: Long conversation (100+ messages)")
    long_messages = []
    for i in range(150):  # Create 150 messages
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i+1}: {' '.join([f'word{j}' for j in range(20)])}"  # Make each message longer
        long_messages.append({"role": role, "content": content})

    # Add system message at the beginning
    long_messages.insert(
        0,
        {
            "role": "system",
            "content": "You are a helpful assistant in a very long conversation.",
        },
    )

    config = TransformConfig(
        enabled=True,
        max_messages=20,  # Very restrictive limit
        preserve_system_messages=True,
        preserve_recent_messages=5,
    )
    transforms = MessageTransforms(config)
    result = transforms.transform_messages(long_messages)

    print(
        f"Long conversation: {result.original_message_count} -> {result.compressed_message_count} messages"
    )
    print(
        f"Token reduction: {result.original_token_count} -> {result.compressed_token_count}"
    )

    # Scenario 2: Token-heavy messages
    print("\nScenario 2: Token-heavy content")
    token_heavy_messages = [
        {"role": "system", "content": "You are analyzing code."},
        {
            "role": "user",
            "content": "Analyze this Python file: " + "x = 1\n" * 500,
        },  # Very long code
        {
            "role": "assistant",
            "content": "This appears to be a Python file that repeatedly assigns 1 to variable x. "
            * 100,
        },
    ]

    config = TransformConfig(
        enabled=True,
        max_tokens=2000,  # Restrictive token limit
        preserve_system_messages=True,
    )
    result = transforms.transform_messages(token_heavy_messages)
    print(
        f"Token-heavy content: {result.original_token_count} -> {result.compressed_token_count} tokens"
    )

    # Scenario 3: Mixed content types
    print("\nScenario 3: Mixed message types")
    mixed_messages = [
        {
            "role": "system",
            "content": "You handle various content types.",
        },
        {
            "role": "user",
            "content": "Process this data: [1, 2, 3, 4, 5] * 50",
        },  # List-like content
        {
            "role": "assistant",
            "content": "I've processed your list data.",
        },
        {
            "role": "user",
            "content": "Now process this dict: {'key': 'value'} * 30",
        },  # Dict-like content
        {
            "role": "assistant",
            "content": "Dictionary processed successfully.",
        },
    ]

    result = transforms.transform_messages(mixed_messages)
    print(
        f"Mixed content: {result.original_message_count} -> {result.compressed_message_count} messages"
    )


def example_6_model_specific_limits():
    """Example 6: Model-specific context limits"""
    print("\n" + "=" * 60)
    print("Example 6: Model-Specific Limits")
    print("=" * 60)

    # Test different models and their limits
    models_and_limits = [
        ("gpt-4", 8192),
        ("gpt-4-turbo", 128000),
        ("claude-3-sonnet", 200000),
        ("claude-2", 100000),
        ("gpt-3.5-turbo", 16385),
    ]

    sample_content = "This is a sample message. " * 100  # ~300 tokens
    messages = [
        {"role": "user", "content": sample_content} for _ in range(10)
    ]

    for model, expected_limit in models_and_limits:
        config = TransformConfig(
            enabled=True,
            model_name=model,
            preserve_system_messages=False,
        )
        transforms = MessageTransforms(config)
        result = transforms.transform_messages(messages)

        print(
            f"{model}: {result.original_token_count} -> {result.compressed_token_count} tokens (limit: {expected_limit})"
        )


def main():
    """Run all examples"""
    print("🚀 Swarms Message Transforms Examples")
    print("=" * 60)

    try:
        example_1_basic_transforms()
        example_2_dictionary_config()
        example_3_agent_integration()
        example_4_convenience_function()
        example_5_advanced_scenarios()
        example_6_model_specific_limits()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("• Transforms automatically handle context size limits")
        print("• Middle-out compression preserves important context")
        print("• System messages and recent messages are prioritized")
        print("• Works with any LLM model through the Agent class")
        print("• Detailed logging shows compression statistics")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
