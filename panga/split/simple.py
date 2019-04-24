import numpy as np
import os
import logging
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import append_fields, drop_fields
from collections import OrderedDict
from scipy.stats import gaussian_kde
from scipy.signal import argrelmax

from fast5_research.fast5_bulk import BulkFast5
from fast5_research.util import get_changes

from panga import iterators
from panga.fileio import readchunkedtsv

from panga.split.base import SplitBase, Read
from panga.util import add_prefix
from panga.algorithm.deltasplit import read_bounds_from_delta

__all__ = ('FixedInterval', 'Fast5Split', 'DeltaSplit', 'RandomData', 'AdaptiveThreshold', 'SummarySplit',)

logger = logging.getLogger(__name__)

class Fast5Split(SplitBase):

    def __init__(self, fast5, channel, with_events=False, with_raw=False, max_time=np.inf):
        """Provide read data by utilising pre-calculated split points stored
        in fast5 file.

        :param fast5: fast5 file.
        :param channel: channel for which to provide reads.
        :param with_events: bool, whether to provide event data.
        :param with_raw: bool, whether to provide raw data.

        """
        super(Fast5Split, self,).__init__()
        self.fast5 = fast5
        self.channel = channel
        self.with_events = with_events
        self.with_raw = with_raw
        self.max_time = max_time

    @property
    def provides_events(self):
        return self.with_events

    @property
    def provides_raw(self):
        return self.with_raw

    @property
    def meta_keys(self):
        return (
            'read_id', 'initial_classification',
            'median_current', 'median_sd',
            'range_current', 'median_dwell', 'drift',
            'start_time', 'duration',
            'num_events', 'start_event', 'end_event',
            'mux', 'states', 'channel', 'mux_changes', 'bias_voltage_changes'
        )

    def load_fast5_meta(self, fh):
        self.channel_meta = fh.get_metadata(self.channel)
        # fetch context meta from the source bulk file
        try:
            self.context_meta = fh.get_context_meta()
        except:
            self.context_meta = {}

        # fetch tracking meta from the source bulk file
        try:
            self.tracking_meta = fh.get_tracking_meta()
        except:
            self.tracking_meta = {'device_id': 'Unknown'}

    def reads(self):
        """Yield `Reads` with various meta data provided by MinKnow."""
        with BulkFast5(self.fast5) as fh:

            # load channel, tracking and context meta so we don't need fast5
            # later to e.g. write fast5 files.
            self.load_fast5_meta(fh)

            # use read classification from the penultimate block of multi-block reads
            for read in fh.get_reads(self.channel, penultimate_class=True):
                event_indices = (read['event_index_start'],
                                 read['event_index_end'])

                read_events = None
                if self.with_events:
                    read_events = fh.get_events(self.channel,
                                                event_indices=event_indices)
                    read_events = self._convert_event_fields(read_events, fh.sample_rate)

                # map new dict keys to read columns
                meta_keys = [
                    ('read_id', 'read_id'),
                    ('initial_classification', 'classification'),
                    ('median_current', 'median'),
                    ('median_sd', 'median_sd'),
                    ('range_current', 'range'),
                    ('median_dwell', 'median_dwell'),
                    ('start_time', 'read_start'),
                    ('duration', 'read_length'),
                    ('drift', 'drift')
                ]
                meta = {key:read[col] for key, col in meta_keys}
                divide = ('median_dwell','duration','start_time')
                for name in divide:
                    meta[name] = float(meta[name]) / fh.sample_rate

                meta.update({
                    'num_events':event_indices[1] - event_indices[0],
                    'start_event':event_indices[0],
                    'end_event':event_indices[1],

                })
                self._add_channel_states(fh, meta)

                if set(meta.keys()) != set(self.meta_keys):
                    extra = set(meta.keys()) - set(self.meta_keys)
                    missing = set(self.meta_keys) - set(meta.keys())
                    raise ValueError(
                        '{} about to yield read with unexpected metrics. '
                        'Extra: {}. Missing {}.'.format(
                        self.__class__, extra, missing)
                    )
                read_raw = None
                if self.with_raw:
                    read_raw = fh.get_raw(self.channel, times=(meta['start_time'], meta['start_time'] + meta['duration']), use_scaling=False)

                if meta['start_time'] > self.max_time:
                    raise StopIteration

                yield Read(events=read_events, raw=read_raw, meta=meta, channel_meta=self.channel_meta,
                           context_meta=self.context_meta, tracking_meta=self.tracking_meta)

    def _add_channel_states(self, fh, meta):
        """Add mux, channel and channel states to meta, with special handling of mux 0.

        For mux 0, find out what last well was (mux 1-4) and look for all states  since when the mux was set to zero.
        If with_mux_changes is True, add a table of all mux-change entries with times, and mux-state values (non-enumerated
        values, thus allowing for the distinction between e.g. 1:common_voltage_1 and 6:unblock_voltage_1), which both enumerate to well_id 1.


        """
        mux = fh.get_mux(self.channel, time=meta['start_time'])
        times = meta['start_time'], meta['start_time'] + meta['duration']
        if mux == 0:  # find out what the last well was, and when it was set. Note the mux could still be zero,
            # if the well was off from the start of the run
            mux, mux_set_time = fh.get_mux(self.channel, time=meta['start_time'], wells_only=True, return_raw_index=True)
            mux_set_time = float(mux_set_time) / fh.sample_rate  # convert from raw index to time
            # look for any channel states which might have caused mux to change to zero (e.g. saturated / multiple),
            # i.e. look in time window from mux change to end of read
            states = fh.get_states_in_window(self.channel, times=(mux_set_time, meta['start_time'] + meta['duration']))
        else:
            states = fh.get_states_in_window(self.channel, times=times)

        mux_changes = fh.get_mux_changes_in_window(self.channel, times=times)
        # ensure 'well_id' changes between rows of the mux_changes struct array
        mux_changes = get_changes(mux_changes, use_cols=('well_id',))
        logger.debug('mux changes from {} to {}: {}'.format(times[0], times[1], mux_changes))
        change_times = mux_changes['approx_raw_index'] / fh.sample_rate
        mux_changes = drop_fields(mux_changes,'approx_raw_index', usemask=False)
        mux_changes = append_fields(mux_changes, 'time', change_times, usemask=False)

        meta.update({
            'mux': mux,
            'states': states,
            'channel': self.channel,
            'mux_changes': mux_changes,
            'bias_voltage_changes': fh.get_bias_voltage_changes_in_window(times=times)
        })


    def _convert_event_fields(self, read_events, sample_rate):
        """Convert event fields 'start' and 'length' from raw indices into times
        """
        # convert event fields 'start' and 'length' from raw indices into times
        for col in ['start', 'length']:
            times = read_events[col] / sample_rate
            read_events = drop_fields(read_events, col, usemask=False)
            read_events = append_fields(read_events, col, times, usemask=False)
        return read_events


class DeltaSplit(Fast5Split):

    def __init__(self, fast5, channel, with_events=True, with_raw=False, delta=90, look_back=3, max_time=np.inf):
        """Provide read data using a delta splitter.

        :param fast5: fast5 file.
        :param channel: channel for which to provide reads.
        :param with_events: bool, whether to provide event data.
        :param with_raw: bool, whether to provide raw data.
        :param delta: change in `mean` defining boundaries
        :param look_back: number of events over which to look for change `delta`
        """
        super(DeltaSplit, self,).__init__(fast5, channel, with_events=with_events, with_raw=with_raw, max_time=max_time)
        self.delta = delta
        self.look_back = look_back

    @property
    def meta_keys(self):
        return (
            'start_time', 'duration', 'start_event', 'end_event', 'num_events',
            'mux', 'states', 'channel', 'mux_changes', 'bias_voltage_changes',
        )

    def reads(self):
        """Yield `Reads` obtained from delta splitting."""

        with BulkFast5(self.fast5) as fh:

            # load channel, tracking and context meta so we don't need fast5
            # later to e.g. write fast5 files.
            self.load_fast5_meta(fh)

            events = fh.get_events(self.channel)

            bounds = read_bounds_from_delta(
                events['mean'], delta=self.delta, look_back_n=self.look_back
            )
            for event_indices in bounds:
                read_events = None
                if self.with_events:
                    read_events = events[event_indices[0]:event_indices[1]]
                    read_events = self._convert_event_fields(read_events, fh.sample_rate)

                meta = {
                    'start_time': read_events[0]['start'],
                    'duration': read_events[-1]['start'] + read_events[-1]['length'] - read_events[0]['start'],
                    'num_events':event_indices[1] - event_indices[0],
                    'start_event':event_indices[0],
                    'end_event':event_indices[1],
                }
                self._add_channel_states(fh, meta)

                if set(meta.keys()) != set(self.meta_keys):
                    extra = set(meta.keys()) - set(self.meta_keys)
                    missing = set(self.meta_keys) - set(meta.keys())
                    raise ValueError(
                        '{} about to yield read with unexpected metrics. '
                        'Extra: {}. Missing {}.'.format(
                        self.__class__, extra, missing)
                    )
                read_raw = None
                if self.with_raw:
                    read_raw = fh.get_raw(self.channel, times=(meta['start_time'], meta['start_time'] + meta['duration']), use_scaling=False)

                if meta['start_time'] > self.max_time:
                    raise StopIteration

                yield Read(events=read_events, raw=read_raw, meta=meta, channel_meta=self.channel_meta,
                           context_meta=self.context_meta, tracking_meta=self.tracking_meta)


class FixedInterval(SplitBase):

    def  __init__(self, fast5, channel, interval=100):
        """Provide read data by chunking a channel into fixed length pieces.

        :param fast5: fast5 file to read.
        :param channel: channel for which to provide reads.
        :param interval: number of events per read.
        """
        super(FixedInterval, self,).__init__()
        self.fast5 = fast5
        self.channel = channel
        self.interval = interval

    @property
    def provides_events(self):
        return True

    @property
    def provides_raw(self):
        return False

    @property
    def meta_keys(self):
        return tuple()

    def reads(self):
        with BulkFast5(self.fast5) as fh:
            # load channel, tracking and context meta so we don't need fast5
            # later to e.g. write fast5 files.
            self.load_fast5_meta(fh)
            for chunk in iterators.blocker(fh.get_events(self.channel), self.interval):
                yield Read(events=chunk)


class RandomData(SplitBase):

    def __init__(self, n_reads=10, read_length=100):
        """Generate random reads.

        :param n_reads: number of reads to produce.
        :param read_length: number of events per read.
        """
        self.n_reads = n_reads
        self.read_length = read_length

    @property
    def provides_events(self):
        return True

    @property
    def provides_raw(self):
        return False

    @property
    def meta_keys(self):
        return tuple()

    def reads(self):
        spread = 1
        start = 0.0
        for mean in np.random.choice(range(20,50,5), size=self.n_reads):
            read = np.empty(self.read_length, dtype=[
                ('mean',float), ('stdv', float), ('start', float), ('length', float)
            ])
            read['mean'] = np.random.normal(mean, spread, size=self.read_length)
            read['stdv'] = 1.0
            read['length'] = 1.0
            read['start'] = np.linspace(start, start + self.read_length, num=self.read_length, endpoint=False)
            yield Read(events=read)
            start += read['start'][-1] + read['length'][-1]


class AdaptiveThreshold(Fast5Split):

    def  __init__(self, fast5, channel, outpath, prefix, with_raw=False, max_time=np.inf, pore_rank=0, capture_rank=1, thresh_factor=0.9, fixed_threshold=None):
        """Split based on open-pore and strand-capture levels inferred from distribution of event levels.

        :param fast5: fast5 file.
        :param channel: channel for which to provide reads.
        :param threshold: current below which to create a read boundary.
        :param outpath: output path for plot.
        :param prefix: prefix (prefixed to output plot)
        :param pore_rank: int, ranking of pore current within kde local maxima,
               defaults corresponds to highest probability peak.
        :param capture_rank: int, ranking of capture current within kde local maxima,
               defaults corresponds to second highest probability peak.
        :param thresh_factor: float, factor determining position of threshold between capture and pore levels. 0.5 corresponds to mid-point.
               threshold = capture_level + thresh_factor * (pore_level - capture_level)
        :param fixed_threshold: float, use this fixed threshold. If None, calculate threshold.
        """
        super(AdaptiveThreshold, self,).__init__(fast5, channel, with_events=True, with_raw=with_raw, max_time=max_time)
        times = (0, max_time) if max_time < np.inf else None
        if fixed_threshold is None:
            self.pore_level, self.capture_level, self.threshold = self._get_levels(outpath, prefix, times, pore_rank, capture_rank, thresh_factor)
        else:
            self.threshold = fixed_threshold
            self.pore_level, self.capture_level = None, None

    @property
    def provides_events(self):
        return True

    @property
    def provides_raw(self):
        return self.with_raw

    @property
    def meta_keys(self):
        return (
            'start_time', 'duration', 'start_event', 'end_event', 'num_events',
            'mux', 'states', 'channel', 'mux_changes', 'pore_level',
            'capture_level', 'threshold',
            'bias_voltage_changes',
        )

    def reads(self):
        with BulkFast5(self.fast5) as fh:

            # load channel, tracking and context meta so we don't need fast5
            # later to e.g. write fast5 files.
            self.load_fast5_meta(fh)

            events = fh.get_events(self.channel)
            # convert event fields 'start' and 'length' from raw indices into times
            for col in ['start', 'length']:
                times = events[col] / fh.sample_rate
                events = drop_fields(events, col, usemask=False)
                events = append_fields(events, col, times, usemask=False)

            read_bound_event_indices = np.where(np.ediff1d((events['mean'] < self.threshold).astype(int)) !=0)[0]
            # first event should be start of first read
            read_bound_event_indices = np.insert(read_bound_event_indices + 1, 0, 0)
            # pad end with last event index + 1.
            read_bound_event_indices = np.append(read_bound_event_indices, len(events) - 1)
            for start_event, next_start_event in iterators.window(read_bound_event_indices, 2):
                read_events = events[start_event: next_start_event]
                start_t, end_t = events[start_event]['start'], events[next_start_event]['start']
                meta = {
                    'start_time': start_t,
                    'duration': end_t - start_t,
                    'pore_level': self.pore_level,
                    'capture_level': self.capture_level,
                    'threshold': self.threshold,
                    'start_event': start_event,
                    'end_event': next_start_event,
                    'num_events': len(read_events),
                }
                self._add_channel_states(fh, meta)
                read_raw = None
                if self.with_raw:
                    read_raw = fh.get_raw(self.channel, times=(start_t, end_t), use_scaling=False)

                if meta['start_time'] > self.max_time:
                    raise StopIteration

                yield Read(events=read_events, raw=read_raw, meta=meta, channel_meta=self.channel_meta,
                           context_meta=self.context_meta, tracking_meta=self.tracking_meta)

    def _get_levels(self, outpath, prefix, times=None, pore_rank=0, capture_rank=1, thresh_factor=0.9):
        """Calculate distribution of event means, and infer open-pore level and  capture level.

        Assumes the pore level corresoponds to the highest-probability peak in
        the distribution, and that the capture level is the second highest.

        :param outpath: directory in which to plot the distribution and levels.
        :param prefix: prefix (prefixed to output plot path)
        :param times: (start time, end time) or None
        :param pore_rank: int, ranking of pore current within kde local maxima,
               defaults corresponds to highest probability peak.
        :param capture_rank: int, ranking of capture current within kde local maxima,
               defaults corresponds to second highest probability peak.
        :param thresh_factor: float, factor f with which to calculate boundary threshold;
                threshold = capture_level + f * (pore_level - capture_level)
                a value of 0.5 implies the midpoint between pore and capture.
        :returns: tuple of floats, (pore_level, capture_level, threshold)
        """
        with BulkFast5(self.fast5) as fh:
            logger.info('Loading events for channel {}'.format(self.channel))
            events = fh.get_events(self.channel, times=times)

        logger.info('Calculating kde for channel {}'.format(self.channel))
        kde = gaussian_kde(events['mean'], bw_method='silverman')  # silverman is seemingly better for multi-modal dists
        logger.info('Done calculating kde for channel {}'.format(self.channel))
        x = np.linspace(np.min(events['mean']), np.max(events['mean']), 100)

        pde_vals = kde(x)  # evaluate density over grid
        max_inds = argrelmax(kde(x))  # find all local maxima
        max_probs = pde_vals[max_inds]
        sorted_inds = np.argsort(max_probs)[::-1]  # so max prob is 1st elem
        pore_ind = max_inds[0][sorted_inds[pore_rank]]
        capture_ind = max_inds[0][sorted_inds[capture_rank]]

        pore_level = x[pore_ind]
        capture_level = x[capture_ind]
        threshold = capture_level + thresh_factor * (pore_level - capture_level)

        # plot kde, histogram and levels.
        fig, axis = plt.subplots()
        axis.hist(events['mean'], bins=100, color='k', label='histogram')
        axis.legend(loc='upper center', frameon=False)
        axis.set_xlim((-100, 400))
        axis2 = axis.twinx()

        axis2.plot(x, kde(x), label='kde', color='k')
        axis2.plot(x[max_inds], pde_vals[max_inds],'o', label='local maxima', color='b')
        axis2.plot(x[pore_ind], pde_vals[pore_ind],'o', label='open pore current', color='r')
        axis2.plot(x[capture_ind], pde_vals[capture_ind],'o', label='capture current', color='g')
        axis.axvline(threshold, label='threshold', color='magenta')
        axis2.legend(loc='upper left', frameon=False)
        plot_path = os.path.join(outpath, add_prefix('AdaptiveThresholdLevels_{}'.format(self.channel, prefix)))
        plt.savefig(plot_path, bbox_inches='tight', dpi=200)
        with open(plot_path + '.txt', 'w') as fh:
            fh.write('#pore rank {}\n'.format(pore_rank))
            fh.write('#capture rank {}\n'.format(capture_rank))
            fh.write('#thresh_factor {}\n'.format(thresh_factor))
            fh.write('#pore level {}\n'.format(pore_level))
            fh.write('#capture level {}\n'.format(capture_level))
            fh.write('#threshold level {}\n'.format(threshold))
            # write local maxima in kde distribution
            fh.write('# probability maxima in kde \n')
            fh.write('\t'.join(['pA', 'kde']) + '\n')
            for i in range(len(max_probs)):
                j = max_inds[0][sorted_inds[i]]
                fh.write('\t'.join(map(str, [x[j], pde_vals[j]])) + '\n')
            # write sampled kde
            fh.write('# kde points \n')
            fh.write('\t'.join(['pA', 'kde']) + '\n')
            for xi, yi in zip(x, pde_vals):
               fh.write('\t'.join(map(str, [xi, yi])) + '\n')

        return pore_level, capture_level, threshold


class SummarySplit(Fast5Split):

    def __init__(self, input_summary, channel, fast5=None, with_events=False, with_raw=False, with_states=False, max_time=np.inf):
        """Provide read data by utilising pre-calculated split points stored
        in a summary file.

        :param input_summary: summary file.
        :param channel: channel for which to provide reads.
        :param fast5: fast5 file, only needed if with_events, with_raw or with_states is True.
        :param with_events: bool, whether to provide event data from the fast5
        :param with_raw: bool, whether to provide raw data from the fast5.
        :param with_states: if True, add channel state data (mux and channel states) from the fast5.

        """
        super(SummarySplit, self,).__init__(fast5, channel, with_events=with_events, with_raw=with_raw, max_time=max_time)
        self.input_summary = input_summary
        self.with_states = with_states

        if (self.with_events or self.with_raw or self.with_states) and self.fast5 is None:
            raise RuntimeError('Raw, events or channel_states requested, but no fast5 provided')

        with open(self.input_summary, 'r') as fh:
            col_names = tuple(fh.readline().strip().split('\t'))
            col_vals = tuple(fh.readline().strip().split('\t'))
            self._input_summary_cols = self.__guess_dtype(OrderedDict(zip(col_names, col_vals)))

        if not set(self.requires_meta).issubset(self._input_summary_cols.keys()):
            missing = set(self.requires_meta) - set(self._input_summary_cols.keys())
            raise KeyError('Input summary does not provide all read meta required '
                ': Missing {}'.format(missing))

    def __guess_dtype(self, read_metrics):
        """Guess dtype of columns from dict of col name : value pairs"""

        dtypes = OrderedDict()
        for name, val in read_metrics.items():
            if 'is_' in name:  # is_saturated, is_multiple etc
                dtypes[name] = lambda x: bool(int(x))
            elif 'current' in name:  # could be current_before, which for first read might be 0 (int)
                dtypes[name] = float
            else:
                try:
                    _ = int(val)
                    dtypes[name] = int
                except:
                    try:
                        _ = float(val)
                        dtypes[name] = float
                    except:
                        dtypes[name] = str
        return dtypes

    def iterate_input_file(self):
        for i, chunk in enumerate(readchunkedtsv(self.input_summary, 10000)):
            logger.info('Read chunk {} ({} reads) of input summary'.format(i, len(chunk)))
            names = chunk.dtype.names
            for rec in (dict(zip(names, x)) for x in chunk):
                if rec['channel'] != self.channel:
                    # we only want to process reads for a single channel
                    # but this means looping over the file 512 times.. might
                    # there be a better way to do this?
                    continue
                yield rec

    @property
    def requires_meta(self):
        return (
            'start_time',
            'duration',
            'channel'
        )

    @property
    def meta_keys(self):
        if self.with_states:
            return tuple(['mux', 'states', 'channel', 'mux_changes', 'bias_voltage_changes'] + self._input_summary_cols.keys() )
        else:
            return tuple(self._input_summary_cols.keys())

    def reads(self) :
        """Yield `Reads` with various meta data provided by MinKnow."""

        if self.with_events or self.with_raw or self.with_states:
            f5 = BulkFast5(self.fast5)
            # load channel, tracking and context meta so we don't need fast5
            # later to e.g. write fast5 files.
            self.load_fast5_meta(f5)
        else:
            # initialise fast5/channel meta variables so we have a generic call
            # to Read constructor even when we don't have the fast5
            self.channel_meta = None
            self.context_meta = None
            self.tracking_meta = None

        if self.with_events:
            if set(['start_event', 'end_event']).issubset(self.meta_keys):
                get_events = lambda meta: f5.get_events(self.channel, event_indices=(meta['start_event'], meta['end_event']))
            else:
                logger.warn('Reading events using timings, this will be slow.')
                get_events = lambda meta: f5.get_events(self.channel, times=(meta['start_time'], meta['start_time'] + meta['duration']))

        for meta in self.iterate_input_file():
            read_events = None
            if self.with_events:
                read_events = get_events(meta)
                read_events = self._convert_event_fields(read_events, f5.sample_rate)
            read_raw = None
            if self.with_raw:
                read_raw = f5.get_raw(self.channel, times=(meta['start_time'], meta['start_time'] + meta['duration']), use_scaling=False)
            if self.with_states:  # add mux, channel states from fast5
                self._add_channel_states(f5, meta)

            if meta['start_time'] > self.max_time:
                raise StopIteration

            yield Read(events=read_events, raw=read_raw, meta=meta, channel_meta=self.channel_meta,
                       context_meta=self.context_meta, tracking_meta=self.tracking_meta)

        if self.with_events or self.with_raw or self.with_states:
            f5.close()
