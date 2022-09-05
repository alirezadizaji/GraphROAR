from contextlib import contextmanager
import time

from main import global_seed
from ..context.constants import Constants

@contextmanager
def seed_changer(end=100000):
    seed = int(time.time()) % end
    print(f"@@@@@ seed changes to {seed} @@@@@", flush=True)
    global_seed(seed)        
    yield
    global_seed(Constants.GLOBAL_SEED)
    