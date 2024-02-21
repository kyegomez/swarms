import subprocess

from swarms.telemetry.check_update import check_for_update


def auto_update():
    """auto update swarms"""
    try:
        if check_for_update is True:
            print(
                "There is a new version of swarms available!"
                " Downloading..."
            )
            subprocess.run(["pip", "install", "--upgrade", "swarms"])
        else:
            print("swarms is up to date!")
    except Exception as e:
        print(e)
