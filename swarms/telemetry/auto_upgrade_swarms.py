import subprocess

from swarms.telemetry.check_update import check_for_update
from termcolor import colored


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
            colored("swarms is up to date!", "red")
    except Exception as e:
        print(e)
