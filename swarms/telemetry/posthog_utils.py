import logging

from dotenv import load_dotenv
from posthog import Posthog


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PosthogWrapper:
    def __init__(
        self, api_key, instance_address, debug=False, disabled=False
    ):
        self.posthog = Posthog(api_key, host=instance_address)
        self.posthog.debug = debug
        self.posthog.disabled = disabled

    def capture_event(self, distinct_id, event_name, properties=None):
        self.posthog.capture(distinct_id, event_name, properties)

    def capture_pageview(self, distinct_id, url):
        self.posthog.capture(
            distinct_id, "$pageview", {"$current_url": url}
        )

    def set_user_properties(
        self, distinct_id, event_name, properties
    ):
        self.posthog.capture(
            distinct_id, event=event_name, properties=properties
        )

    def is_feature_enabled(
        self, flag_key, distinct_id, send_feature_flag_events=True
    ):
        return self.posthog.feature_enabled(
            flag_key, distinct_id, send_feature_flag_events
        )

    def get_feature_flag_payload(self, flag_key, distinct_id):
        return self.posthog.get_feature_flag_payload(
            flag_key, distinct_id
        )

    def get_feature_flag(self, flag_key, distinct_id):
        return self.posthog.get_feature_flag(flag_key, distinct_id)

    def capture_with_feature_flag(
        self, distinct_id, event_name, flag_key, variant_key
    ):
        self.posthog.capture(
            distinct_id,
            event_name,
            {"$feature/{}".format(flag_key): variant_key},
        )

    def capture_with_feature_flags(
        self, distinct_id, event_name, send_feature_flags=True
    ):
        self.posthog.capture(
            distinct_id,
            event_name,
            send_feature_flags=send_feature_flags,
        )
