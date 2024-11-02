import os
import sentry_sdk
import threading


os.environ["USE_TELEMETRY"] = "True"


def activate_sentry_async():
    use_telementry = os.getenv("USE_TELEMETRY")

    if use_telementry == "True":
        sentry_sdk.init(
            dsn="https://5d72dd59551c02f78391d2ea5872ddd4@o4504578305490944.ingest.us.sentry.io/4506951704444928",
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            enable_tracing=True,
            debug=False,  # Set debug to False
        )


def activate_sentry():
    t = threading.Thread(target=activate_sentry_async)
    t.start()


# if __name__ == "__main__":
#     run_in_new_thread(activate_sentry)
