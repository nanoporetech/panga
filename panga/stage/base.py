import abc
import copy
from collections import defaultdict
import itertools

from panga import iterators


class StageBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, splitter, classifier):
        """Generic interface for manipulating Reads and metric streams."""
        # TODO: maybe will want to enforce properties of reads and metrics
        #       from supplied splitter and classifier

    @abc.abstractmethod
    def _process_reads(self, reads, metrics):
        """Analyse a stream of reads, however you like, yielding a tuple
        of generators (read, metrics).
        """

    def process_reads(self, reads, metrics):
        """Manipulate a stream of reads and associated metrics."""
        # return two new iterators, one each over new reads and new metrics
        new_reads, new_metrics = self._process_reads(reads, metrics)
        return new_reads, new_metrics
