import gc
from typing import Callable, Sequence

import torch


def _garbage_collect():
    """Garbage collect both CUDA and RAM."""
    torch.cuda.empty_cache()
    gc.collect()


def garbage_collect(func: Callable | None = None) -> Callable | None:
    """Either garbage collects or decorates a function with garbage collection.

    When called with no arguments, garbage collect both CUDA memory and CPU
    memory immeditately. If provided with a function as an argument,
    it will act as a decorator and return a function which garbage collects
    before and after being called.

    Decorator usage:

        @garbage_collect
        def f():
            ...

    Args:
        func: A function to decorate

    Returns:
        None if called without func. With func, the same function that
            garbage collects before and after being called.
    """
    if func is None:
        _garbage_collect()
    else:

        def f(*args, **kwargs):
            _garbage_collect()
            r = func(*args, **kwargs)
            _garbage_collect()
            return r

        return f


def split(seq: Sequence, chunksize: int) -> list[Sequence]:
    """Split a sequence into chunks of size chunksize.

    If the sequence isn't evenly divisible by chunksize, the last chunk
    will be smaller.

    Args:
        seq: The sequence to split
        chunksize: The size of each chunk

    Returns:
        A list of sequences of size chunksize
    """
    return [seq[i : i + chunksize] for i in range(0, len(seq), chunksize)]
