from itertools import tee
from typing import Iterable

def pairwise(iterable: Iterable) -> Iterable:
    """Pairwise implementation in case of using > python3.10.

    Based on official implementation: 
    https://docs.python.org/3/library/itertools.html#itertools.pairwise

    Args:
        iterable (Iterable): Any iterable object
    Returns:
        Iterable: pairwise('ABCDEFG') --> AB BC CD DE EF FG 
    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)