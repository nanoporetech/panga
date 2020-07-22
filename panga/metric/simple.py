import numpy as np

from functools import partial
from panga.metric.base import MetricBase
from panga.metric.metrics import apply_to_field, duration, range_data, get_channel_attr, get_channel_attr_at_read, get_read_meta_attr, read_start_time, is_saturated_at_read, is_multiple_at_read, is_off_at_read, n_global_flicks_in_read, n_active_flicks_in_read, windowed_metric, entropy, locate_stall
from panga.iterators import window

__all__ = ('SimpleMetrics', 'MockMetrics', 'StandardMetrics', 'StandardMinionMetrics', 'StandardMinknowMetrics', 'SummaryMetrics', 'SolidStateMetrics', 'BlockingMetrics', 'SummaryAndStandardMetrics')


class SimpleMetrics(MetricBase):
    def __init__(self, splitter):
        """Example of registering functions to calculate metrics."""
        self.register_event_metric('mean', apply_to_field(np.mean, 'mean'))
        self.register_event_metric('stdv', apply_to_field(np.std, 'mean'))
        super(SimpleMetrics, self).__init__(splitter)


class MockMetrics(MetricBase):
    def __init__(self, *metrics):
        """Mock object useful for testing. Variadic args are used as provided
        metrics (the associated function simple returns its input, the read).
        """
        for metric in metrics:
            self.register_event_metric(metric, lambda x: x)


class StandardMetrics(MetricBase):
    def __init__(self, splitter, channel_meta=None):
        """Calculate a standard set of metrics."""
        self.register_event_metric('median_current', apply_to_field(np.median, 'mean'))
        self.register_event_metric('median_sd', apply_to_field(np.median, 'stdv'))
        self.register_event_metric('median_dwell', apply_to_field(np.median, 'length'))
        self.register_event_metric('num_events', len)
        self.register_event_metric('start_time', read_start_time)
        self.register_event_metric('duration', duration)
        self.register_event_metric('range_current', apply_to_field(range_data, 'mean'))
        super(StandardMetrics, self).__init__(splitter, channel_meta=channel_meta)


class StandardMinionMetrics(StandardMetrics):
    def __init__(self, splitter, channel_meta=None):
        """Calculate a standard set of metrics including Minion state data."""
        self.register_read_meta_metric('states', 'is_saturated', is_saturated_at_read)
        self.register_read_meta_metric('states', 'is_multiple', is_multiple_at_read)
        self.register_read_meta_metric('bias_voltage_changes', 'n_global_flicks', n_global_flicks_in_read)
        self.register_read_meta_metric('mux_changes', 'n_active_flicks', n_active_flicks_in_read)

        self.register_read_meta_metric('mux', 'is_off', is_off_at_read)
        self.register_read_meta_metric('mux')
        self.register_read_meta_metric('channel')
        self.register_read_meta_metric('start_event')
        self.register_read_meta_metric('end_event')
        super(StandardMinionMetrics, self).__init__(splitter, channel_meta=channel_meta)


class StandardMinknowMetrics(MetricBase):
    def __init__(self, splitter, channel_meta=None):
        """Calculate a standard set of metrics including Minion state data. Do
        not calculate metrics from scratch but rather copy from MinKNOW
        provided read meta data.

        """
        # add same metrics as StandardMinionMetrics but without deriving from StandardMetrics
        self.register_read_meta_metric('states', 'is_saturated', is_saturated_at_read)
        self.register_read_meta_metric('states', 'is_multiple', is_multiple_at_read)
        self.register_read_meta_metric('bias_voltage_changes', 'n_global_flicks', n_global_flicks_in_read)
        self.register_read_meta_metric('mux_changes', 'n_active_flicks', n_active_flicks_in_read)
        self.register_read_meta_metric('mux', 'is_off', is_off_at_read)
        self.register_read_meta_metric('mux')
        self.register_read_meta_metric('channel')
        # copy read meta as metrics
        copy_meta = (
            'median_current', 'median_sd', 'median_dwell', 'num_events',
            'start_time', 'duration', 'range_current', 'drift',
            'start_event', 'end_event', 'initial_classification',
        )
        for name in copy_meta:
            self.register_read_meta_metric(name)
        super(StandardMinknowMetrics, self).__init__(splitter, channel_meta=channel_meta)


class SummaryMetrics(MetricBase):
    def __init__(self, splitter, channel_meta=None):
        """Use read_meta provided by the splitter as metrics.
        """
        # copy read meta as metrics
        copy_meta = splitter.meta_keys
        for name in copy_meta:
            self.register_read_meta_metric(name)
        super(SummaryMetrics, self).__init__(splitter, channel_meta=channel_meta)


class SolidStateMetrics(StandardMinionMetrics):
    def __init__(self, splitter, channel_meta=None):
        """Calculate metrics of interest for solid state runs (but potentially useful elsewhere"""

        self.register_event_metric('max_current', apply_to_field(np.max, 'mean'))  # as estimate of abasic peak
        self.register_event_metric('stdev_current', apply_to_field(np.std, 'mean'))

        # copy read meta from AdaptiveSplitter as metrics
        copy_meta = ('pore_level', 'capture_level', 'threshold')
        for name in copy_meta:
            self.register_read_meta_metric(name)
        #TODO: add metric for strand level (excluding abasic), for whole read, and up to start of abasic peak.
        super(SolidStateMetrics, self).__init__(splitter, channel_meta=channel_meta)


class BlockingMetrics(StandardMinionMetrics):
    def __init__(self, splitter, channel_meta=None, window_bounds=(0,1,3,500,900)):
        """Calculate a standard set of metrics including Minion state data."""
        copy_meta = (
            'start_event', 'end_event', 'initial_classification', 'initial_n_reads',
        )
        for name in copy_meta:
            self.register_read_meta_metric(name)
        self.register_event_metric('entropy_current', apply_to_field(entropy, 'mean'))

        self.register_event_multi_metric(
            ('stall_duration', 'stall_median_current', 'stall_range'), locate_stall
        )

        # register mean current calculated in several windows
        for start, end in window(window_bounds, 2):
            self.register_event_metric(
                'median_current_{}_to_{}_s'.format(start, end),
                 partial(windowed_metric, feature='mean',
                     window_start=start, window_length=end-start, func_to_apply=np.median)
            )
        super(BlockingMetrics, self).__init__(splitter, channel_meta=channel_meta)


class SummaryAndStandardMetrics(SummaryMetrics, StandardMetrics):
    pass

