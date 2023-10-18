# disable warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# disable tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# swarms

# from swarms.chunkers import chunkers
from swarms.logo import logo

print(logo)
