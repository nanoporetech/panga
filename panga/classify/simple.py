from panga.classify.base import ClassifyBase

__all__ = ('SimpleClassifier', 'DeltaClassifier', 'StandardClassifier', 'MetricClassifier')


class SimpleClassifier(ClassifyBase):

    def __init__(self, metrifier):
        """Example classifier which simply examines the metric `mean`
        to determine the class of a read.
        """
        super(SimpleClassifier, self).__init__(metrifier)

    @property
    def n_reads(self):
        return 1

    @property
    def requires(self):
        return ['median_current']

    def _process_read(self, metrics):
        # Note the [0] here because the interface dictates that we get passed
        #   a tuple of metrics for a window of reads.
        if metrics[0]['median_current'] > 150:
            return 'pore'
        else:
            return 'not_pore'


class DeltaClassifier(ClassifyBase):
    """Classify reads according to mean current before, during, and after."""

    def __init__(self, metrifier):
        super(DeltaClassifier, self).__init__(metrifier)

    @property
    def n_reads(self):
        return 3

    @property
    def requires(self):
        return ['mean']

    def _process_read(self, metrics):
        delta0 = 'down' if metrics[0]['mean'] - metrics[1]['mean'] > 0 else "up"
        delta1 = 'down' if metrics[1]['mean'] - metrics[2]['mean'] > 0 else "up"
        return "{}-{}".format(delta0, delta1)


class StandardClassifier(ClassifyBase):
    def __init__(self, metrifier, class_metric='initial_classification',
                 to_inject=(('median_current', -1, '_before', 0.0),
                            ('median_current', 1, '_after', 0.0)),
                 time_since=('strand',), recovered_classes=None,
                 state_classes=None,
                 ):
        """Set read class from an existing metric.

        :param class_metric: str, key for read metric used as class.
        :param to_inject metrics: iterable of tuples of (key, offset, suffix, fill)
            where key is the metric key, offset is the int offset relative to
            the central read, the str suffix is appended to the metric key, and fill
            is the value assigned if the metrics of the proceeding read are None
            (which might happen for the first/last read).
        :param time_since: str, if not None, inject time since the last read
            classed with the provided class filter.
        :param recovered_classes: iterable of str, if not None, inject boolean asserting
            whether the read class of the next read is in recovered_classes.
        :param state_classes: iterable of str, if not None, inject state metrics
            e.g. n_<class>_local for each class in state_classes. See ClassifyBase for details.

        ..note:: if read metrics contain 'is_saturated==True' this will
            override the classification scheme.

        .. note::
            to_inject=(('median_current', -1, '_before', 0.0)) would inject the new
            metric 'median_current_before' with the value of the
            'median_current' of the preceeding read.
        """

        self.class_metric = class_metric
        self.to_inject = to_inject
        self.inject_time_since = time_since
        self.recovered_classes = recovered_classes
        self.state_classes = self.recovered_classes
        super(StandardClassifier, self).__init__(metrifier)

    @property
    def n_reads(self):
        return 3

    @property
    def requires(self):
        requires = ['is_saturated', 'is_multiple', 'is_off']
        if self.class_metric is not None:  # could be None in derived classes
            requires.append(self.class_metric)
        if self.recovered_classes is not None:
            requires.append('mux')
        # update requires with metrics to inject
        requires = self._add_inject_requires(requires)
        return requires

    def _process_read(self, metrics):
        klass = 'unclassed'
        # if channel is off, change read class
        if 'is_off' in metrics[self.central_read] and metrics[self.central_read]['is_off']:
            klass = 'off'
        # If it is in the multiple pore state, no matter what,
        if 'is_multiple' in metrics[self.central_read] and metrics[self.central_read]['is_multiple']:
            klass = 'multiple'
        # If it is saturated, no matter what (overwriting any potential multiple state)
        if 'is_saturated' in metrics[self.central_read] and metrics[self.central_read]['is_saturated']:
            klass = 'saturated'
        if klass not in ['saturated', 'multiple', 'off']:
            klass = self._classify_read(metrics)
        return klass

    def _classify_read(self, metrics):
        return str(metrics[self.central_read][self.class_metric])

    def _add_inject_requires(self, requires):
        """Update requires list with base names of metrics to be injected"""
        metric_names = [t[0] for t in self.to_inject]
        requires = list(set(requires).union(set(metric_names)))
        return requires


class MetricClassifier(StandardClassifier):
    """Set read class from an existing metric without any is_satured checks."""

    @property
    def requires(self):
        requires = []
        if self.class_metric is not None:  # could be None in derived classes
            requires.append(self.class_metric)
        if self.recovered_classes is not None:
            requires.append('mux')
        # update requires with metrics to inject
        requires = self._add_inject_requires(requires)
        return requires

    def _process_read(self, metrics):
        return self._classify_read(metrics)
