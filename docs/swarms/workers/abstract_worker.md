# AbstractWorker Class
====================

The `AbstractWorker` class is an abstract class for AI workers. An AI worker can communicate with other workers and perform actions. Different workers can differ in what actions they perform in the `receive` method.

## Class Definition
----------------

```
class AbstractWorker:
    """(In preview) An abstract class for AI worker.

    An worker can communicate with other workers and perform actions.
    Different workers can differ in what actions they perform in the `receive` method.
    """
```


## Initialization
--------------

The `AbstractWorker` class is initialized with a single parameter:

-   `name` (str): The name of the worker.

```
def __init__(
    self,
    name: str,
):
    """
    Args:
        name (str): name of the worker.
    """
    self._name = name
```


## Properties
----------

The `AbstractWorker` class has a single property:

-   `name`: Returns the name of the worker.

```
@property
def name(self):
    """Get the name of the worker."""
    return self._name
```


## Methods
-------

The `AbstractWorker` class has several methods:

### `run`

The `run` method is used to run the worker agent once. It takes a single parameter:

-   `task` (str): The task to be run.

```
def run(
    self,
    task: str
):
    """Run the worker agent once"""
```


### `send`

The `send` method is used to send a message to another worker. It takes three parameters:

-   `message` (Union[Dict, str]): The message to be sent.
-   `recipient` (AbstractWorker): The recipient of the message.
-   `request_reply` (Optional[bool]): If set to `True`, the method will request a reply from the recipient.

```
def send(
    self,
    message: Union[Dict, str],
    recipient: AbstractWorker,
    request_reply: Optional[bool] = None
):
    """(Abstract method) Send a message to another worker."""
```


### `a_send`

The `a_send` method is the asynchronous version of the `send` method. It takes the same parameters as the `send` method.

```
async def a_send(
    self,
    message: Union[Dict, str],
    recipient: AbstractWorker,
    request_reply: Optional[bool] = None
):
    """(Abstract async method) Send a message to another worker."""
```


### `receive`

The `receive` method is used to receive a message from another worker. It takes three parameters:

-   `message` (Union[Dict, str]): The message to be received.
-   `sender` (AbstractWorker): The sender of the message.
-   `request_reply` (Optional[bool]): If set to `True`, the method will request a reply from the sender.

```
def receive(
    self,
    message: Union[Dict, str],
    sender: AbstractWorker,
    request_reply: Optional[bool] = None
):
    """(Abstract method) Receive a message from another worker."""
```


### `a_receive`

The `a_receive` method is the asynchronous version of the `receive` method. It takes the same parameters as the `receive` method.

```
async def a_receive(
    self,
    message: Union[Dict, str],
    sender: AbstractWorker,
    request_reply: Optional[bool] = None
):
    """(Abstract async method) Receive a message from another worker."""
```


### `reset`

The `reset` method is used to reset the worker.

```
def reset(self):
    """(Abstract method) Reset the worker."""
```


### `generate_reply`

The `generate_reply` method is used to generate a reply based on the received messages. It takes two parameters:

-   `messages` (Optional[List[Dict]]): A list of messages received.
-   `sender` (AbstractWorker): The sender of the messages.

The method returns a string, a dictionary, or `None`. If `None` is returned, no reply is generated.

```
def generate_reply(
    self,
    messages: Optional[List[Dict]] = None,
    sender: AbstractWorker,
    **kwargs,
) -> Union[str, Dict, None]:
    """(Abstract method) Generate a reply based on the received messages.

    Args:
        messages (list[dict]): a list of messages received.
        sender: sender of an Agent instance.
    Returns:
        str or dict or None: the generated reply. If None, no reply is generated.
    """
```


### `a_generate_reply`

The `a_generate_reply` method is the asynchronous version of the `generate_reply` method. It

takes the same parameters as the `generate_reply` method.

```
async def a_generate_reply(
    self,
    messages: Optional[List[Dict]] = None,
    sender: AbstractWorker,
    **kwargs,
) -> Union[str, Dict, None]:
    """(Abstract async method) Generate a reply based on the received messages.

    Args:
        messages (list[dict]): a list of messages received.
        sender: sender of an Agent instance.
    Returns:
        str or dict or None: the generated reply. If None, no reply is generated.
    """
```


Usage Examples
--------------

### Example 1: Creating an AbstractWorker

```
from swarms.worker.base import AbstractWorker

worker = AbstractWorker(name="Worker1")
print(worker.name)  # Output: Worker1
```


In this example, we create an instance of `AbstractWorker` named "Worker1" and print its name.

### Example 2: Sending a Message

```
from swarms.worker.base import AbstractWorker

worker1 = AbstractWorker(name="Worker1")
worker2 = AbstractWorker(name="Worker2")

message = {"content": "Hello, Worker2!"}
worker1.send(message, worker2)
```


In this example, "Worker1" sends a message to "Worker2". The message is a dictionary with a single key-value pair.

### Example 3: Receiving a Message

```
from swarms.worker.base import AbstractWorker

worker1 = AbstractWorker(name="Worker1")
worker2 = AbstractWorker(name="Worker2")

message = {"content": "Hello, Worker2!"}
worker1.send(message, worker2)

received_message = worker2.receive(message, worker1)
print(received_message)  # Output: {"content": "Hello, Worker2!"}
```


In this example, "Worker1" sends a message to "Worker2". "Worker2" then receives the message and prints it.

Notes
-----

-   The `AbstractWorker` class is an abstract class, which means it cannot be instantiated directly. Instead, it should be subclassed, and at least the `send`, `receive`, `reset`, and `generate_reply` methods should be overridden.
-   The `send` and `receive` methods are abstract methods, which means they must be implemented in any subclass of `AbstractWorker`.
-   The `a_send`, `a_receive`, and `a_generate_reply` methods are asynchronous methods, which means they return a coroutine that can be awaited using the `await` keyword.
-   The `generate_reply` method is used to generate a reply based on the received messages. The exact implementation of this method will depend on the specific requirements of your application.
-   The `reset` method is used to reset the state of the worker. The exact implementation of this method will depend on the specific requirements of your application.