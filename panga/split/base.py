import abc
from uuid import uuid4
from numbers import Integral


class Read(object):
    def __init__(self, events=None, raw=None, meta=None, channel_meta=None,
                 context_meta=None, tracking_meta=None):
        """A type to enforce definition of a read. Split objects should yield
        instances of these. Could probably be a namedtuple.
        """
        if raw is not None:
            required_channel_meta = set(('range', 'digitisation', 'offset'))
            if required_channel_meta is None:
                raise RuntimeError('If providing raw, channel_meta must also be provided')
            elif not required_channel_meta.issubset(channel_meta.keys()):
                raise KeyError('If providing raw, meta must contain {}'.format(', '.join(required_channel_meta)))
            if not isinstance(raw[0], Integral):
                raise TypeError('Raw should be provided as ADC integer.')

        if meta is None:
            meta = {}

        self.events = events
        self.adc_raw = raw  # this should be integer unscaled raw
        self.meta = meta
        self.channel_meta = channel_meta
        self.context_meta = context_meta
        self.tracking_meta = tracking_meta

        if 'read_id' not in meta:
            self.meta['read_id'] = str(uuid4())

    @property
    def raw(self):
        scaled_raw = None
        if self.adc_raw is not None:
            raw_unit = self.channel_meta['range'] / self.channel_meta['digitisation']
            scaled_raw = (self.adc_raw + self.channel_meta['offset']) * raw_unit
        return scaled_raw


class SplitBase(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        """Take input to pipeline to yield `Reads`."""
        pass

    @abc.abstractmethod
    def reads(self):
        """Yield read data."""
        pass

    @abc.abstractproperty
    def provides_events(self):
        pass

    @abc.abstractproperty
    def provides_raw(self):
        pass

    @abc.abstractproperty
    def meta_keys(self):
        pass
