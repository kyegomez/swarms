import functools
import os

from dotenv import load_dotenv
from posthog import Posthog
from swarms.telemetry.user_utils import generate_unique_identifier

# Load environment variables
load_dotenv()


# Initialize Posthog client
def init_posthog(debug: bool = True, *args, **kwargs):
    """Initialize Posthog client.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to True.

    """
    api_key = os.getenv("POSTHOG_API_KEY")
    host = os.getenv("POSTHOG_HOST")
    posthog = Posthog(api_key, host=host, *args, **kwargs)

    if debug:
        posthog.debug = True

    return posthog


def log_activity_posthog(event_name: str, **event_properties):
    """Log activity to Posthog.


    Args:
        event_name (str): Name of the event to log.
        **event_properties: Properties of the event to log.

    Examples:
        >>> from swarms.telemetry.posthog_utils import log_activity_posthog
        >>> @log_activity_posthog("test_event", test_property="test_value")
        ... def test_function():
        ...     print("Hello, world!")
        >>> test_function()
        Hello, world!
        >>> # Check Posthog dashboard for event "test_event" with property
        >>> # "test_property" set to "test_value".
    """

    def decorator_log_activity(func):
        @functools.wraps(func)
        def wrapper_log_activity(*args, **kwargs):
            result = func(*args, **kwargs)

            # Assuming you have a way to get the user id
            distinct_user_id = generate_unique_identifier()

            # Capture the event
            init_posthog.capture(
                distinct_user_id, event_name, event_properties
            )

            return result

        return wrapper_log_activity

    return decorator_log_activity


@log_activity_posthog(
    "function_executed", function_name="my_function"
)
def my_function():
    # Function logic here
    return "Function executed successfully!"


out = my_function()
print(out)
