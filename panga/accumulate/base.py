import abc


class AccumulateBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, class_filter=None):
        """Accumulate read_metrics, supporting context managed use.

        :param class_filter: restrict accumulation to defined read classes.
        """
        self.class_filter = class_filter

    @abc.abstractmethod
    def _process_read(self, read_metrics):
        pass

    def process_read(self, read_metrics):
        """Accumulate a single set of read metrics."""
        if read_metrics is not None and (self.class_filter is None or read_metrics['class'] in self.class_filter):
            self._process_read(read_metrics)

    def process_reads(self, reads):
        """Accumulate an iterable of read metrics."""
        for read_metrics in reads:
            yield self.process_read(read_metrics)

    def finalize(self):
        """Cleanup resources. Clients can call this method to properly cleanup
        resources such as filehandles or network connections."""
        pass

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        self.finalize()
