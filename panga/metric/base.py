import abc

class MetricBase(object):
    __metaclass__ = abc.ABCMeta

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.event_metrics = {}
        obj.event_multi_metrics = []
        obj.channel_metrics = {}
        obj.read_meta_metrics = []
        obj.channel_meta = None
        return obj

    def __init__(self, splitter, channel_meta=None):
        """Calculate a dictionary of read metrics from `Read` objects`.

        :param splitter: a `panga.split.base.SplitBase`. Used on initialization
            to check that splitter will provide necessary data.
        :param channel_meta: deprecated.
        """
        # Derived classes should call this constructor after registering metrics
        self.channel_meta = channel_meta

        if self.requires_events and not splitter.provides_events:
            raise AttributeError('Splitter does not provide events, required by Metrifier.')
        if self.requires_raw and not splitter.provides_raw:
            raise AttributeError('Splitter does not provide raw, required by Metrifier.')
        if not set(self.requires_meta).issubset(splitter.meta_keys):
            missing = set(self.requires_meta) - set(splitter.meta_keys)
            raise KeyError('Splitter does not provide all read meta required '
                'by Metrifier: Missing {}'.format(missing))

        # We will always copy the read_id
        self.register_read_meta_metric('read_id')

    @property
    def provides(self):
        """List of metrics which we compute for a read."""
        #FIXME/TODO; the way event_multi_metrics provides metric names might want to be looked at.
        event_multi_metrics = []
        for x in self.event_multi_metrics:
            for y in x[0]:
                event_multi_metrics.append(y)
        return self.event_metrics.keys() + event_multi_metrics + self.channel_metrics.keys() + [x[1] for x in self.read_meta_metrics]

    @property
    def requires_meta(self):
        """List of read meta keys we expect."""
        return [x[0] for x in self.read_meta_metrics]

    @property
    def requires_events(self):
        """Do we require event data in `Read`s."""
        return len(self.event_metrics.keys()) > 0

    @property
    def requires_raw(self):
        """Do we require raw data in `Read`s."""
        return False

    def process_read(self, read):
        """Process a read."""
        return self._calculate_metrics(read)

    def process_reads(self, reads):
        """Yield metrics for a set of reads."""
        for read in reads:
            yield self.process_read(read)

    def register_event_metric(self, name, func):
        """Register a function to apply to read event data to produce a metric.

        :param name: name of metric.
        :param func: function to calculate metric. Should accept a single
            argument: the `.event` attribute of a read.
        """
        self.event_metrics[name] = func

    def register_event_multi_metric(self, names, func):
        """Register a function to apply to read event data to produce a set of
        metrics.

        :param name: name of metric.
        :param func: function to calculate metric. Should accept a single
            argument: the `.event` attribute of a read.
        """
        self.event_multi_metrics.append((names, func))

    def register_channel_metric(self, name, func):
        """Register a function to apply to channel data.

        :param name: name of metric.
        :param func: function to calculate metric. Should accept two
            arguments: `channel_meta` stored in this class, and `.meta`
            attribute of a read.
        """
        self.channel_metrics[name] = func

    def register_read_meta_metric(self, name, resultant=None, func=None):
        """Copy meta data from read (optionally apply a function and update name.

        :param name: name of metric to copy/use
        :param resultant: resultant name of metric
        :param func: function to apply to original value of metric
        """
        if resultant is None:
            resultant = name
        self.read_meta_metrics.append((name, resultant, func))

    def _calculate_metrics(self, read):
        """Apply metrics to a read to compute a metric dictionary."""
        metrics = dict()
        if read.events is not None:
            metrics.update({
                name:func(read.events) for name, func in self.event_metrics.items()
            })
            for names, func in self.event_multi_metrics:
                metrics.update(dict(zip(
                    names, func(read.events)
                )))
        if self.channel_meta is not None:
            metrics.update({
                name:func(self.channel_meta, read.meta) for name, func in self.channel_metrics.items()
            })
        if read.meta is not None:
            keys, values = [], []
            for name, resultant, func in self.read_meta_metrics:
                keys.append(resultant)
                if func is None:
                    value = read.meta[name]
                else:
                    value = func(read.meta[name])
                values.append(value)
            metrics.update({k:v for k,v in zip(keys, values)})
        return metrics
