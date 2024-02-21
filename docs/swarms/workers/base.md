# `AbstractWorker` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Abstract Worker](#abstract-worker)
    1. [Class Definition](#class-definition)
    2. [Attributes](#attributes)
    3. [Methods](#methods)
3. [Tutorial: Creating Custom Workers](#tutorial-creating-custom-workers)
4. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Swarms library, a powerful tool for building and simulating swarm architectures. This library provides a foundation for creating and managing autonomous workers that can communicate, collaborate, and perform various tasks in a coordinated manner.

In this documentation, we will cover the `AbstractWorker` class, which serves as the fundamental building block for creating custom workers in your swarm simulations. We will explain the class's architecture, attributes, and methods in detail, providing practical examples to help you understand how to use it effectively.

Whether you want to simulate a team of autonomous robots, a group of AI agents, or any other swarm-based system, the Swarms library is here to simplify the process and empower you to build complex simulations.

---

## 2. Abstract Worker <a name="abstract-worker"></a>

### 2.1 Class Definition <a name="class-definition"></a>

The `AbstractWorker` class is an abstract base class that serves as the foundation for creating worker agents in your swarm simulations. It defines a set of methods that should be implemented by subclasses to customize the behavior of individual workers.

Here is the class definition:

```python
class AbstractWorker:
    def __init__(self, name: str):
        """
        Args:
            name (str): Name of the worker.
        """

    @property
    def name(self):
        """Get the name of the worker."""

    def run(self, task: str):
        """Run the worker agent once."""

    def send(
        self, message: Union[Dict, str], recipient, request_reply: Optional[bool] = None
    ):
        """Send a message to another worker."""

    async def a_send(
        self, message: Union[Dict, str], recipient, request_reply: Optional[bool] = None
    ):
        """Send a message to another worker asynchronously."""

    def receive(
        self, message: Union[Dict, str], sender, request_reply: Optional[bool] = None
    ):
        """Receive a message from another worker."""

    async def a_receive(
        self, message: Union[Dict, str], sender, request_reply: Optional[bool] = None
    ):
        """Receive a message from another worker asynchronously."""

    def reset(self):
        """Reset the worker."""

    def generate_reply(
        self, messages: Optional[List[Dict]] = None, sender=None, **kwargs
    ) -> Union[str, Dict, None]:
        """Generate a reply based on received messages."""

    async def a_generate_reply(
        self, messages: Optional[List[Dict]] = None, sender=None, **kwargs
    ) -> Union[str, Dict, None]:
        """Generate a reply based on received messages asynchronously."""
```

### 2.2 Attributes <a name="attributes"></a>

- `name (str)`: The name of the worker, which is set during initialization.

### 2.3 Methods <a name="methods"></a>

Now, let's delve into the methods provided by the `AbstractWorker` class and understand their purposes and usage.

#### `__init__(self, name: str)`

The constructor method initializes a worker with a given name.

**Parameters:**
- `name (str)`: The name of the worker.

**Usage Example:**

```python
worker = AbstractWorker("Worker1")
```

#### `name` (Property)

The `name` property allows you to access the name of the worker.

**Usage Example:**

```python
worker_name = worker.name
```

#### `run(self, task: str)`

The `run()` method is a placeholder for running the worker. You can customize this method in your subclass to define the specific actions the worker should perform.

**Parameters:**
- `task (str)`: A task description or identifier.

**Usage Example (Customized Subclass):**

```python
class MyWorker(AbstractWorker):
    def run(self, task: str):
        print(f"{self.name} is performing task: {task}")


worker = MyWorker("Worker1")
worker.run("Collect data")
```

#### `send(self, message: Union[Dict, str], recipient, request_reply: Optional[bool] = None)`

The `send()` method allows the worker to send a message to another worker or recipient. The message can be either a dictionary or a string.

**Parameters:**
- `message (Union[Dict, str])`: The message to be sent.
- `recipient`: The recipient worker or entity.
- `request_reply (Optional[bool])`: If `True`, the sender requests a reply from the recipient. If `False`, no reply is requested. Default is `None`.

**Usage Example:**

```python
worker1 = AbstractWorker("Worker1")
worker2 = AbstractWorker("Worker2")

message = "Hello, Worker2!"
worker1.send(message, worker2)
```

#### `a_send(self, message: Union[Dict, str], recipient, request_reply: Optional[bool] = None)`

The `a_send()` method is an asynchronous version of the `send()` method, allowing the worker to send messages asynchronously.

**Parameters:** (Same as `send()`)

**Usage Example:**

```python
import asyncio


async def main():
    worker1 = AbstractWorker("Worker1")
    worker2 = AbstractWorker("Worker2")

    message = "Hello, Worker2!"
    await worker1.a_send(message, worker2)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

#### `receive(self, message: Union[Dict, str], sender, request_reply: Optional[bool] = None)`

The `receive()` method allows the worker to receive messages from other workers or senders. You can customize this method in your subclass to define how the worker handles incoming messages.

**Parameters:**
- `message (Union[Dict, str])`: The received message.
- `sender`: The sender worker or entity.
- `request_reply (Optional[bool])`: Indicates whether a reply is requested. Default is `None`.

**Usage Example (Customized Subclass):**

```python
class MyWorker(AbstractWorker):
    def receive(self, message: Union[Dict, str], sender, request_reply: Optional[bool] = None):
        if isinstance(message, str):
            print(f"{self.name} received a text message from {sender.name}: {message}")
        elif isinstance(message, dict):
            print(f"{self.name} received a dictionary message from {sender.name}: {message}")

worker1 = MyWorker("Worker1")
worker2 = MyWorker("Worker2")

message1 =

 "Hello, Worker2!"
message2 = {"data": 42}

worker1.receive(message1, worker2)
worker1.receive(message2, worker2)
```

#### `a_receive(self, message: Union[Dict, str], sender, request_reply: Optional[bool] = None)`

The `a_receive()` method is an asynchronous version of the `receive()` method, allowing the worker to receive messages asynchronously.

**Parameters:** (Same as `receive()`)

**Usage Example:**

```python
import asyncio


async def main():
    worker1 = AbstractWorker("Worker1")
    worker2 = AbstractWorker("Worker2")

    message1 = "Hello, Worker2!"
    message2 = {"data": 42}

    await worker1.a_receive(message1, worker2)
    await worker1.a_receive(message2, worker2)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

#### `reset(self)`

The `reset()` method is a placeholder for resetting the worker. You can customize this method in your subclass to define how the worker should reset its state.

**Usage Example (Customized Subclass):**

```python
class MyWorker(AbstractWorker):
    def reset(self):
        print(f"{self.name} has been reset.")


worker = MyWorker("Worker1")
worker.reset()
```

#### `generate_reply(self, messages: Optional[List[Dict]] = None, sender=None, **kwargs) -> Union[str, Dict, None]`

The `generate_reply()` method is a placeholder for generating a reply based on received messages. You can customize this method in your subclass to define the logic for generating replies.

**Parameters:**
- `messages (Optional[List[Dict]])`: A list of received messages.
- `sender`: The sender of the reply.
- `kwargs`: Additional keyword arguments.

**Returns:**
- `Union[str, Dict, None]`: The generated reply. If `None`, no reply is generated.

**Usage Example (Customized Subclass):**

```python
class MyWorker(AbstractWorker):
    def generate_reply(
        self, messages: Optional[List[Dict]] = None, sender=None, **kwargs
    ) -> Union[str, Dict, None]:
        if messages:
            # Generate a reply based on received messages
            return f"Received {len(messages)} messages from {sender.name}."
        else:
            return None


worker1 = MyWorker("Worker1")
worker2 = MyWorker("Worker2")

message = "Hello, Worker2!"
reply = worker2.generate_reply([message], worker1)

if reply:
    print(f"{worker2.name} generated a reply: {reply}")
```

#### `a_generate_reply(self, messages: Optional[List[Dict]] = None, sender=None, **kwargs) -> Union[str, Dict, None]`

The `a_generate_reply()` method is an asynchronous version of the `generate_reply()` method, allowing the worker to generate replies asynchronously.

**Parameters:** (Same as `generate_reply()`)

**Returns:**
- `Union[str, Dict, None]`: The generated reply. If `None`, no reply is generated.

**Usage Example:**

```python
import asyncio


async def main():
    worker1 = AbstractWorker("Worker1")
    worker2 = AbstractWorker("Worker2")

    message = "Hello, Worker2!"
    reply = await worker2.a_generate_reply([message], worker1)

    if reply:
        print(f"{worker2.name} generated a reply: {reply}")


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

---

## 3. Tutorial: Creating Custom Workers <a name="tutorial-creating-custom-workers"></a>

In this tutorial, we will walk you through the process of creating custom workers by subclassing the `AbstractWorker` class. You can tailor these workers to perform specific tasks and communicate with other workers in your swarm simulations.

### Step 1: Create a Custom Worker Class

Start by creating a custom worker class that inherits from `AbstractWorker`. Define the `run()` and `receive()` methods to specify the behavior of your worker.

```python
class CustomWorker(AbstractWorker):
    def run(self, task: str):
        print(f"{self.name} is performing task: {task}")

    def receive(
        self, message: Union[Dict, str], sender, request_reply: Optional[bool] = None
    ):
        if isinstance(message, str):
            print(f"{self.name} received a text message from {sender.name}: {message}")
        elif isinstance(message, dict):
            print(
                f"{self.name} received a dictionary message from {sender.name}: {message}"
            )
```

### Step 2: Create Custom Worker Instances

Instantiate your custom worker instances and give them unique names.

```python
worker1 = CustomWorker("Worker1")
worker2 = CustomWorker("Worker2")
```

### Step 3: Run Custom Workers

Use the `run()` method to make your custom workers perform tasks.

```python
worker1.run("Collect data")
worker2.run("Process data")
```

### Step 4: Communicate Between Workers

Use the `send()` method to send messages between workers. You can customize the `receive()` method to define how your workers handle incoming messages.

```python
worker1.send("Hello, Worker2!", worker2)
worker2.send({"data": 42}, worker1)

# Output will show the messages received by the workers
```

### Step 5: Generate Replies

Customize the `generate_reply()` method to allow your workers to generate replies based on received messages.

```python
class CustomWorker(AbstractWorker):
    def generate_reply(
        self, messages: Optional[List[Dict]] = None, sender=None, **kwargs
    ) -> Union[str, Dict, None]:
        if messages:
            # Generate a reply based on received messages
            return f"Received {len(messages)} messages from {sender.name}."
        else:
            return None
```

Now, your custom workers can generate replies to incoming messages.

```python
reply = worker2.generate_reply(["Hello, Worker2!"], worker1)

if reply:
    print(f"{worker2.name} generated a reply: {reply}")
```

---

## 4. Conclusion <a name="conclusion"></a>

Congratulations! You've learned how to use the Swarms library to create and customize worker agents for swarm simulations. You can now build complex swarm architectures, simulate autonomous systems, and experiment with various communication and task allocation strategies.

Feel free to explore the Swarms library further and adapt it to your specific use cases. If you have any questions or need assistance, refer to the extensive documentation and resources available.

Happy swarming!