import os
import sentry_sdk
import threading


os.environ["USE_TELEMETRY"] = "True"


def activate_sentry_async():
    use_telementry = os.getenv("USE_TELEMETRY")

    if use_telementry == "True":
        #sentry_sdk.init(
        #    #dsn="https://5d72dd59551c02f78391d2ea5872ddd4@o4504578305490944.ingest.us.sentry.io/4506951704444928",
        #)
        sentry_sdk.init(
            dsn="https://4fd91d75ad5635da55cdd3069e8fdd97@o4508452173840384.ingest.de.sentry.io/4508452178493520",
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for tracing.
            traces_sample_rate=1.0,
            #traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            enable_tracing=True,
            debug=True,  # Set debug to False

            _experiments={
                # Set continuous_profiling_auto_start to True
                # to automatically start the profiler on when
                # possible.
                "continuous_profiling_auto_start": True,
            },
        )

#asgi_app = SentryAsgiMiddleware(asgi_app)



def activate_sentry():
    t = threading.Thread(target=activate_sentry_async)
    t.start()


# if __name__ == "__main__":
#     run_in_new_thread(activate_sentry)
