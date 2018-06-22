#cython: boundscheck=False
#cython: wraparound=False
# Import the numpy cython module shipped with Cython.
from cpython cimport bool
cimport numpy as np
import numpy as np


def read_bounds_from_delta(np.ndarray[float, ndim=1] pas, double delta, int look_back_n):
    """
    Detects event boundaries using algorithm copied from ossetra.strand_detector
    """

    cdef np.ndarray[int, ndim=1] bounds = np.empty(dtype=np.int32, shape=len(pas)+1)
    cdef np.ndarray[double, ndim=1] look_back = np.empty(dtype=np.float, shape=look_back_n)
    cdef int i,j
    cdef int n_bounds = 0
    cdef int n_cached = 0
    cdef double last_pa = pas[0]
    cdef double look_back_drop = 0.0
    cdef double tmp_drop = 0.0
    cdef bool event = False
    cdef bool lb_event = False

    # start is a boundary
    bounds[0] = 0
    n_bounds += 1

    for i in range(1, len(pas)-1):

        pa = pas[i]
        event = False
        lb_event = False

        ##########################################

        drop = abs(pa - last_pa)

        look_back_drop = 0.0
        # find max look_back_drop
        if n_cached > 0:
            look_back_drop = abs(pa - look_back[0])
            for j in range(1, n_cached):
                tmp_drop = abs(pa - look_back[j])
                if tmp_drop > look_back_drop:
                    look_back_drop = tmp_drop

        if  drop > delta:
            event = True

        elif look_back_drop > delta:
            lb_event = True

        ##########################################

        if event or lb_event:
            bounds[n_bounds] = i
            n_bounds += 1
            n_cached = 0

        if n_cached == look_back_n:
            # 'pop' off first element of look_back
            for j in range(1, n_cached):
                look_back[j-1] = look_back[j]
            n_cached -= 1

        look_back[n_cached] = pa
        n_cached += 1

        last_pa = pa

    # the last event is a bound
    bounds[n_bounds] = len(pas)
    n_bounds += 1

    bounds = bounds[:n_bounds]
    bounds.sort()

    return zip(bounds[:n_bounds-1], bounds[1:n_bounds])

