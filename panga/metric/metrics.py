import functools
import operator
import numpy as np
from scipy.stats import entropy as scipy_entropy

from panga.iterators import window

def _apply_to_field(function, field, obj):
    return function(obj[field])


def apply_to_field(function, field):
    """Create a function which applies a given function to an attribute of an
    object.

    :param function: function to apply, e.g. `np.mean`.
    :param field: attribute of an object.
    """
    return functools.partial(_apply_to_field, function, field)


def duration(events):
    """Calculate total duration of events array.

    :param events: the events to process, should have `start` and
        `length` fields.
    """
    return events[-1]['start'] + events[-1]['length'] - events[0]['start']


def read_start_time(events):
    """Simply return the time of the first event."""
    return events[0]['start']


def range_data(data):
    """Calculate a range measure for a field."""
    centiles = np.percentile(data, (90, 10))
    return centiles[0] - centiles[1]


def med_mad_data(data, axis=None):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median

    :param data: A :class:`ndarray` object
    :param axis: For multidimensional arrays, which axis to calculate over
    :returns: a tuple containing the median and MAD of the data
    """
    dmed = np.median(data, axis=axis)
    dmad = np.median(abs(data - dmed), axis=axis)
    return dmed, dmad


def get_channel_attr(item, *args):
    """Retrieve a pure channel attribute, not dependent on read meta. This
    is only required as metric interface determines the interface."""
    channel_meta = args[0]
    return channel_meta[item]


def get_read_meta_attr(item, read_meta):
    """Retrieve a pure read meta attribute, not dependent on channel meta. This
    is only required as metric interface determines the interface."""
    return read_meta[item]


def get_channel_attr_at_read(item, channel_meta, read_meta):
    """Retrieve a channel attribute at the start time of a read.
    :param item: the meta item to fetch.
    :param channel_meta: dictionary of channel meta.
    :param read_meta: read_meta dictionary with `start_time` field
    """
    start = read_meta['start_time']
    # look for the previous item in channel meta after start.
    # handle special case of start=0
    if start == 0.0:
        i = 0
    else:
        i = np.searchsorted(channel_meta[item]['time'], start) - 1

    return channel_meta[item]['value'][i]


def is_saturated_at_read(states):
    """Retrieve whether the channel is saturated based on the read states.
    :param states: iterable of channel states
    """
    return 'saturated' in states


def is_multiple_at_read(states):
    """Retrieve whether the channel is in the 'multiple' state based on the read states.
    :param states: iterable of channel states
    """
    return 'multiple' in states


def n_global_flicks_in_read(bias_voltage_changes):
    """Retrieve number global that occur within the read.

    :param bias_voltage_changes: structured np array with columns 'time' and 'set_bias_voltage' used to detect global flicks.
    """
    # detect global static flicks, as detected by bias < 100 mV.  from fast5 API
    # sequencing is reported to be at +180, while a flick features a period
    # where the bias is ~ -70 mV. If the bias voltage changes are obtained from
    # the asic commands, there is a change from 180 to -70, while if the bias
    # changes are parsed from the expt history, the bias changes from 180 to 20,
    # then 20 to -70. To handle both cases without double counting, we label
    # everything < 100 mV as belonging to a flick - but this may need revising
    # if sequencing and/or flick voltages change signicantly. To avoid
    # overcounting flicks, we count neighbouring periods in which the voltage is
    # < 100 as 1 flick.
    is_flick = bias_voltage_changes['set_bias_voltage'] < 100
    # create a mask to retrieve only points when we change in and out of flick
    flick_state_changes = np.insert(np.ediff1d(is_flick.astype(int)), 0, [True])
    flick_starts = bias_voltage_changes[np.where(np.logical_and(flick_state_changes, is_flick))]
    n_global_flicks = len(flick_starts)

    return n_global_flicks


def n_active_flicks_in_read(mux_changes):
    """Retrieve number flicks (either global or active) that occur within the read.

    :param mux_changes: structured np array with columns 'time' and 'well_id' used to detect active flicks
    """
    # look for changes in mux to unblock muxes
    unblock_muxes = [6, 7, 8, 9]
    unblock_mux_changes = mux_changes[np.where(np.in1d(mux_changes['well_id'], unblock_muxes))]
    n_active_flicks = len(unblock_mux_changes)

    return n_active_flicks


def is_off_at_read(mux):
    """Retrieve whether the channel is 'off' based on the mux.
    :param mux: int, mux
    """
    return mux == 0


def windowed_metric(events, window_start=0.002, window_length=0.5, feature="mean", func_to_apply=np.mean):
    """Calculate a windowed metric for a region of events, specified by a start-time post failure and a length.

    :param events: the events to process, should have `start` field
    :param window_start: the start of the event region to summarise
    :param window_length: the length of the event region
    :param feature: the event feature to summarise
    """
    start = events[0]['start']
    if((events[-1]['start'] - start) < (window_length + window_start)):
        return np.nan
    window_start_index, window_end_index = np.searchsorted(
        events['start'], [start + window_start, start + window_start + window_length]
    )
    try:
        return_value  = func_to_apply(events[window_start_index:window_end_index][feature])
    except:
        return_value = np.nan
    return (return_value)


def entropy(arr, maximum=600, width=10):
    binned_means = np.digitize(arr, bins=np.arange(maximum, step=width))
    probs = np.bincount(binned_means) / float(len(binned_means))
    return scipy_entropy(probs)


def threshold_changepoint(arr, threshold, min_samples=3, direction='down'):
    """Locate end of high (or low if `direction=='up'`) contiguous entries at
    the beginning of an array.

    :param arr: input array
    :param threshold: threshold above (below) which a changepoint is detected
    :param min_samples: minimum number of entries we need to see under (above)
        the threshold to determine a boundary.
    """
    if direction == 'down':
        op1 = operator.le
        op2 = operator.gt
    elif direction == 'up':
        op1 = operator.ge
        op2 = operator.lt
    else:
        raise ValueError('Direction must be one of "down" or "up".')

    # For when the stall starts below (above) the threshold:
    count_above = 0
    start_ev_ind = 0
    for ev_ind, item in enumerate(arr[:100]):
        if op1(item, threshold):
            count_above = 0
        else:
            count_above += 1

        if count_above == 2:
            start_ev_ind = ev_ind - 1
            break

    new_start = 0
    count = 0
    for idx in range(start_ev_ind, len(arr)):
        if op2(arr[idx], threshold):
            count = 0
        else:
            count += 1

        if count == min_samples:
            # find the time the first event went below - taking just the number
            # away gets the last time *above* the threshold, so add 1
            new_start = idx - min_samples + 1
            break

    return new_start


def locate_stall(events, mad_estimate_events=100):
    """Locate a stall (abnormally high currents) at beginning of read.

    :param events: event data.
    :param mad_estimate_events: number of events at end of read to use to
       estimate median and median absolute deviation.
    """
    med, mad = med_mad_data(events['mean'][-mad_estimate_events:])
    max_thresh = med + 2 * 1.48 * mad
    stall_end = threshold_changepoint(events['mean'], max_thresh)

    if stall_end > 0:
        stall_events = events[:stall_end]
        stall_length = np.sum(stall_events['length'])
        stall_median = np.median(stall_events['mean'])
        stall_range = range_data(stall_events['mean'])
    else:
        stall_length, stall_median, stall_range = 0, 0, 0

    return stall_length, stall_median, stall_range


def filled_window(data, up, down, fill):
    """Create a moving window across input data with boundary windows
    padded with fill value.

    :param up: higher offset index position
    :param down: lower offset index position, should be negative to indicate
        position prior to the considered position.
    :param fill: value to use to fill boundaries.

    """
    wlen = up - down + 1
    for i in xrange(abs(down)):
        s = abs(down)-i
        yield np.concatenate(([fill]*s, data[:wlen-s]))
    for w in window(data, wlen):
        yield np.array(w)
    for i in xrange(abs(up)):
        s = abs(up)-i
        yield np.concatenate((data[-(wlen-s):], [fill]*s))


def sliding_metric(data, metric, up, down, fill):
    """Calculate metric over moving window across input data where boundary
    windows are padded with fill value.

    :param data: array-like
    :param metric: metric to be applied over the window
    :param up: higher offset index position
    :param down: lower offset index position, should be negative to indicate
        position prior to the considered position
    :param fill: value to use to fill boundaries

    :returns: array of length data

    .. note: where there are undefined window elements either side of central index
        (i.e. at the ends of the array) window is padded with fill prior to applying metric
    """
    return np.array([metric(window) for window in filled_window(
        data, up, down, fill)])
