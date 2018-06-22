import abc
import os
import sys

class ConcludeBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, class_filter=None):
        """Do something with a reads given `Read`s and read metrics. Can be
        used within a context manager.

        :param class_filter: restrict to processing defined classes of reads.
        """
        self.class_filter = class_filter

    @abc.abstractmethod
    def _process_read(self, read, read_metrics):
        pass

    def process_read(self, read, read_metrics):
        """Process a single `Read` with metrics."""
        if self.class_filter is None or read_metrics['class'] in self.class_filter:
            self._process_read(read, read_metrics)

    def process_reads(self, reads):
        """Process an iterable of `Read` ad read metrics pairs."""
        for read, read_metrics in reads:
            yield self.process_read(read, read_metrics)

    def finalize(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        self.finalize()

