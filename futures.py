import concurrent.futures
import time
import random
from swarms.utils.futures import execute_futures_dict


def f(x):
    time.sleep(random.random())
    return x


with concurrent.futures.ThreadPoolExecutor() as executor:
    """Create a dictionary of futures."""
    fs_dict = {str(i): executor.submit(f, i) for i in range(10)}
    print(execute_futures_dict(fs_dict))
