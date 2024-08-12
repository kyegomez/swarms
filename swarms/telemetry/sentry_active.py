import os
from dotenv import load_dotenv
import sentry_sdk

load_dotenv()

os.environ["USE_TELEMETRY"] = "True"

use_telementry = os.getenv("USE_TELEMETRY")


def activate_sentry():
    if use_telementry == "True":
        sentry_sdk.init(
            dsn="https://5d72dd59551c02f78391d2ea5872ddd4@o4504578305490944.ingest.us.sentry.io/4506951704444928",
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            enable_tracing=True,
            debug=True,
        )
