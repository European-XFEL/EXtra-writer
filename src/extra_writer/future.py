import sys
from packaging import version

# compatibility to itertools
if sys.version_info < (3, 8):
    import itertools

    def accumulate(iterable, *, initial=None):
        """Return running totals"""
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total += element
            yield total

    itertools.accumulate = accumulate


# compatibility to future numpy features
import numpy as np

if version.parse(np.version.version) < version.parse("1.20"):

    from numpy.core.overrides import set_module

    def _broadcast_shape(*args):
        """Returns the shape of the arrays that would result from broadcasting the
        supplied arrays against each other.
        """
        # use the old-iterator because np.nditer does not handle size 0 arrays
        # consistently
        b = np.broadcast(*args[:32])
        # unfortunately, it cannot handle 32 or more arguments directly
        for pos in range(32, len(args), 31):
            # ironically, np.broadcast does not properly handle np.broadcast
            # objects (it treats them as scalars)
            # use broadcasting to avoid allocating the full array
            b = np.broadcast_to(0, b.shape)
            b = np.broadcast(b, *args[pos:(pos + 31)])
        return b.shape

    @set_module('numpy')
    def broadcast_shapes(*args):
        """Broadcast the input shapes into a single shape."""
        arrays = [np.empty(x, dtype=[]) for x in args]
        return _broadcast_shape(*arrays)

    np.broadcast_shapes = broadcast_shapes
