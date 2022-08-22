from contextlib import contextmanager
import time

from ...main import global_seed


@contextmanager
def seed_changer(end=100000):
    prev_seed = global_seed.seed
    seed = int(time.time()) % end

    global_seed(seed)        
    yield
    global_seed(prev_seed)
    