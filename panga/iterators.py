import numpy as np
from itertools import tee, chain


def window(iterable, size):
    """Create an iterator returning a sliding window from another iterator

    :param iterable: Iterator
    :param size: Size of window

    :returns: an iterator returning a tuple containing the data in the window

    """
    assert size > 0, "Window size for iterator should be strictly positive, got {0}".format(size)
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def blocker(iterable, n):
    """Yield successive n-sized blocks from iterable
       as numpy array. Doesn't pad final block.
    """
    for i in range(0, len(iterable), n):
        yield np.array(iterable[i:i + n])


def empty_iterator(it):
    """Check if an iterator is empty and prepare a fresh one for use

    :param it: iterator to test

    :returns: bool, iterator
    """
    it, any_check = tee(it)
    try:
        next(any_check)
    except StopIteration:
        return True, it
    else:
        return False, it
