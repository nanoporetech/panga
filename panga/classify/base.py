import abc
import copy
from collections import defaultdict

from panga import iterators


class ClassifyBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, metrifier):
        """Compute the class of a read using metrics derived from the read,
        and potentially surrounding reads.
        """
        if self.n_reads % 2 != 1:
            raise ValueError('Classifiers must consume an odd number of reads')
        if not set(self.requires).issubset(metrifier.provides):
            raise KeyError('Metric creator does not provide all metrics '
                           'required by classifier. Missing: {}'.format(
                            set(self.requires) - set(metrifier.provides)))
        self.central_read = self.n_reads // 2

    @abc.abstractproperty
    def n_reads(self):
        """The number of reads required to compute the class."""
        pass

    @abc.abstractproperty
    def requires(self):
        """Metrics required to compute the class. Useful when building a
        pipeline to ensure the metrics that will be passed contain the
        required metrics.
        """
        pass

    @abc.abstractmethod
    def _process_read(self, metrics):
        """Calculate the class of a read given a metric dictionary."""
        pass

    def process_read(self, metrics):
        """Compute class of a central read given metrics for surrounding reads.

        :param metrics: a tuple of metric dictionaries.

        :returns: metric dictionary for read updated with `class` entry.
        """
        assert len(metrics) == self.n_reads
        klass = self._process_read(metrics)
        # return a copy because the original may be used for subsequent reads
        read_metrics = copy.copy(metrics[len(metrics) // 2])
        read_metrics['class'] = klass
        # inject neighbouring metrics if this is required
        if hasattr(self, 'to_inject'):
            self.__inject_neighbouring_metrics(read_metrics, metrics)
        # inject time since last strand
        if hasattr(self, 'inject_time_since') and self.inject_time_since is not None:
            self.__inject_time_since(read_metrics)
        # inject is_recovered
        if hasattr(self, 'recovered_classes') and self.recovered_classes is not None:
            self.__inject_is_recovered(read_metrics, metrics)
        if hasattr(self, 'state_classes') and self.state_classes is not None:
            self.__inject_state_metrics(read_metrics)
        return read_metrics

    def process_reads(self, metrics):
        """Compute classes for a set of reads given metrics for all reads.

        :param metrics: iterable of metric dictionaries for all reads in order.
        """
        padded = iterators.chain(
            [None] * (self.n_reads // 2),
            metrics,
            [None] * (self.n_reads // 2)
        )
        window_metrics = iterators.window(padded, self.n_reads)
        for win_metrics in window_metrics:
            yield self.process_read(win_metrics)

    def __inject_neighbouring_metrics(self, read_metrics, metrics):
        """Inject metrics from neighboring reads into the centre read metrics.

        :param read_metrics: metric dict to inject into.
        :param metrics: a tuple of metric dictionaries to inject from.
        :param to_inject metrics: iterable of tuples of (key, offset, suffix, fill)
            where key is the metric key, offset is the int offset relative to
            the central read, the str suffix is appended to the metric key, and fill
            is the value assigned if the metrics of the proceeding read are None
            (which might happen for the first/last read).

        .. note::
            to_inject=(('median_current', -1, '_before', 0)) would inject the new
            metric 'median_current_before' with the value of the
            'median_current' of the preceeding read.
        """
        for col, offset, suffix, fill in self.to_inject:
            index = self.central_read + offset
            if metrics[index] is not None:
                read_metrics[col + suffix] = metrics[index][col]
            else:
                read_metrics[col + suffix] = fill

    def __inject_time_since(self, read_metrics):
        """Inject time since the previous read with a specific class (e.g. strand).

        :param read_metrics: metric dict to inject into.

        """
        # set up default dict to contain last strand end time for each class and channel
        if not hasattr(self, 'prev_end_time'):
            self.prev_end_time = defaultdict(lambda :defaultdict(float)) # [channel][klass]

        channel = read_metrics['channel']
        for klass in self.inject_time_since:
            metric_name = 'time_since_' + klass
            read_metrics[metric_name] = read_metrics['start_time'] - self.prev_end_time[channel][klass]

        if read_metrics['class'] in self.inject_time_since:
            self.prev_end_time[channel][read_metrics['class']] = read_metrics['start_time'] + read_metrics['duration']

    def __inject_is_recovered(self, read_metrics, metrics):
        """Inject metric asserting whether the next read 'initial_classification' is in self.recovered_classes,
        returning False if the next read is on another mux, or this read is the last read.
        """
        is_recovered = False
        next_read = metrics[self.central_read + 1]
        if (next_read is not None  # this is not last read in channel
            and next_read['mux'] == read_metrics['mux']  # we are on same mux
            and next_read['initial_classification'] in self.recovered_classes
           ):
            is_recovered = True

        read_metrics['is_recovered'] = is_recovered

    def __inject_state_metrics(self, read_metrics):
        """Inject metrics requiring caching of data, e.g. accumulative number/
        duration of reads in local and global contexts."""
        contexts = ('local', 'global')
        state_metrics = {'n': int,
                         'duration': float,
                         'mean_duration': float,
                         }
        # set up counters for cumulative counts and durations
        if not hasattr(self, '_state_data'):
            self._state_data = {}
            for context in contexts:
                self._state_data[context] = {}
                for st_metric, init in state_metrics.items():
                    self._state_data[context][st_metric] = \
                        defaultdict(lambda :defaultdict(lambda: defaultdict(init)))  # per klass/ch/mux

        klass = read_metrics['initial_classification']
        ch = read_metrics['channel']
        mux = read_metrics['mux']

        # inject metrics
        to_inject = {}
        for st_klass in self.state_classes:
            for st_metric in state_metrics.keys():
                for context in contexts:
                    name = '_'.join((st_metric, st_klass, context))
                    to_inject[name] = self._state_data[context][st_metric][st_klass][ch][mux]
        read_metrics.update(to_inject)

        # increment counters
        if klass in self.state_classes:
            for context in contexts:
                self._state_data[context]['n'][klass][ch][mux] += 1
                self._state_data[context]['duration'][klass][ch][mux] += read_metrics['duration']
                self._state_data[context]['mean_duration'][klass][ch][mux] = float(
                    self._state_data[context]['duration'][klass][ch][mux] /
                    float(self._state_data[context]['n'][klass][ch][mux])
                )  # FIXME/TODO: this seems to be coming out as int sometimes

        # reset counters
        elif klass == 'unproductive':
            for st_klass in self.state_classes:
                for st_metric, init in state_metrics.items():
                    self._state_data['local'][st_metric][st_klass][ch][mux] = init()

