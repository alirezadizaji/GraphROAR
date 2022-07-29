from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
import os
import time
import traceback
from typing import Callable

def stdout_stderr_setter(save_dir: str) -> Callable:
    """ it returns a decorator whose objective is to return a function setting the stdout and stderr system objects
    
    Args:
        save_dir (str): the directory where the output and error system files are stored.
    
    Returns:
        decorate (Callable): a decorator that returns a function setting the stdout, sterr objects
    """   

    def decorate(function: Callable) -> Callable:
        """ a decorator returning a new function whose objective is to set the stdout and stderr objects. """
            
        @wraps(function)
        def run_function(*args, **kwargs):
            os.makedirs(save_dir, exist_ok=True)
        
            new_stdout = os.sep.join([save_dir, "console_%d_o.log"])
            new_stderr = os.sep.join([save_dir, "console_%d_e.log"])

            console_num = 0
            while os.path.exists(new_stdout % console_num):
                console_num += 1
            
            print('Consoles are being saved in ', new_stdout % console_num, ' & ', new_stderr % console_num, 'w')

            with open(new_stdout % console_num, 'w') as stdout_:
                with redirect_stdout(stdout_):

                    with open(new_stderr % console_num, 'w') as stderr_:
                        with redirect_stderr(stderr_):
                            start = time.time()
                            try:
                                function(*args, **kwargs)
                                stat = 'SUCCESSFULLY'
                            except:
                                print(f"**ABORT: checkout the following\n***\n{traceback.format_exc()}\n***", flush=True, file=stderr_)
                                stat = 'with ERROR'
                            finally:
                                duration = time.time() - start
                                print(f'FINISHED {stat} in {duration}s.', flush=True)
        return run_function
    
    return decorate