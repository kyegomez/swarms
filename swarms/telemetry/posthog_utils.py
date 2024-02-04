import logging

from dotenv import load_dotenv
from posthog import Posthog


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PosthogWrapper:
    """
    A wrapper class for interacting with the PostHog analytics service.

    Args:
        api_key (str): The API key for accessing the PostHog instance.
        instance_address (str): The address of the PostHog instance.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        disabled (bool, optional): Whether to disable tracking. Defaults to False.
    """

    def __init__(
        self, api_key, instance_address, debug=False, disabled=False
    ):
        self.posthog = Posthog(api_key, host=instance_address)
        self.posthog.debug = debug
        self.posthog.disabled = disabled

    def capture_event(self, distinct_id, event_name, properties=None):
        """
        Capture an event in PostHog.

        Args:
            distinct_id (str): The distinct ID of the user.
            event_name (str): The name of the event.
            properties (dict, optional): Additional properties associated with the event. Defaults to None.
        """
        self.posthog.capture(distinct_id, event_name, properties)

    def capture_pageview(self, distinct_id, url):
        """
        Capture a pageview event in PostHog.

        Args:
            distinct_id (str): The distinct ID of the user.
            url (str): The URL of the page.
        """
        self.posthog.capture(
            distinct_id, "$pageview", {"$current_url": url}
        )

    def set_user_properties(
        self, distinct_id, event_name, properties
    ):
        """
        Set user properties in PostHog.

        Args:
            distinct_id (str): The distinct ID of the user.
            event_name (str): The name of the event.
            properties (dict): The user properties to set.
        """
        self.posthog.capture(
            distinct_id, event=event_name, properties=properties
        )

    def is_feature_enabled(
        self, flag_key, distinct_id, send_feature_flag_events=True
    ):
        """
        Check if a feature flag is enabled for a user.

        Args:
            flag_key (str): The key of the feature flag.
            distinct_id (str): The distinct ID of the user.
            send_feature_flag_events (bool, optional): Whether to send feature flag events. Defaults to True.

        Returns:
            bool: True if the feature flag is enabled, False otherwise.
        """
        return self.posthog.feature_enabled(
            flag_key, distinct_id, send_feature_flag_events
        )

    def get_feature_flag_payload(self, flag_key, distinct_id):
        """
        Get the payload of a feature flag for a user.

        Args:
            flag_key (str): The key of the feature flag.
            distinct_id (str): The distinct ID of the user.

        Returns:
            dict: The payload of the feature flag.
        """
        return self.posthog.get_feature_flag_payload(
            flag_key, distinct_id
        )

    def get_feature_flag(self, flag_key, distinct_id):
        """
        Get the value of a feature flag for a user.

        Args:
            flag_key (str): The key of the feature flag.
            distinct_id (str): The distinct ID of the user.

        Returns:
            str: The value of the feature flag.
        """
        return self.posthog.get_feature_flag(flag_key, distinct_id)

    def capture_with_feature_flag(
        self, distinct_id, event_name, flag_key, variant_key
    ):
        """
        Capture an event with a feature flag in PostHog.

        Args:
            distinct_id (str): The distinct ID of the user.
            event_name (str): The name of the event.
            flag_key (str): The key of the feature flag.
            variant_key (str): The key of the variant.

        """
        self.posthog.capture(
            distinct_id,
            event_name,
            {"$feature/{}".format(flag_key): variant_key},
        )

    def capture_with_feature_flags(
        self, distinct_id, event_name, send_feature_flags=True
    ):
        """
        Capture an event with all feature flags in PostHog.

        Args:
            distinct_id (str): The distinct ID of the user.
            event_name (str): The name of the event.
            send_feature_flags (bool, optional): Whether to send feature flags. Defaults to True.
        """
        self.posthog.capture(
            distinct_id,
            event_name,
            send_feature_flags=send_feature_flags,
        )
