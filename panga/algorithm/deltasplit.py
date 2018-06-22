def read_bounds_from_delta(pas, delta=90, look_back_n=3, use_cython=True):
    if use_cython:
        import read_bounds_from_delta
        # Minkow changes from float64 to float32, and read_bounds_from_delta is cython code
        if pas.dtype != 'float32':
            import numpy as np
            pas = pas.astype(np.float32)
        bounds = read_bounds_from_delta.read_bounds_from_delta(pas, delta, look_back_n)
    else:
        bounds = _read_bounds_from_delta(pas, delta=delta, look_back_n=look_back_n)
    return bounds


def _read_bounds_from_delta(pas, delta=90, look_back_n=3):
    """
    Detects event boundaries using algorithm copied from ossetra.strand_detector
    """
    # first event is a boundary
    bounds = [0]

    look_back = []
    last_pa = pas[0]

    for i in range(1, len(pas)-1):

        pa = pas[i]
        event = False
        lb_event = False

        ##########################################

        drop = abs(pa - last_pa)

        look_back_drop = 0
        if look_back:
            look_back_drop = max([abs(pa-x) for x in look_back])

        if  drop > delta:
            event = True

        elif look_back_drop > delta:
            lb_event = True

        ##########################################

        if event or lb_event:
            bounds.append(i)
            look_back = []

        if len(look_back) >= look_back_n:
            look_back.pop(0)

        look_back.append(pa)

        last_pa = pa

    # last event is a boundary
    bounds.append(len(pas))

    bounds = sorted(bounds)

    return zip(bounds[:-1], bounds[1:])


