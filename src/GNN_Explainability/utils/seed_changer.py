from contextlib import contextmanager
import time

from main import global_seed


@contextmanager
def seed_changer(end=100000):
    seed = int(time.time()) % end

    global_seed(seed)        
    yield
    # BUG: reverting seed is hard code
    global_seed(12345)
    