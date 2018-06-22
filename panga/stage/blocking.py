import itertools
import logging

from fast5_research.fast5_bulk import BulkFast5

from panga.stage.base import StageBase
from panga.split.base import Read
from panga.split import Fast5Split
from panga.metric import BlockingMetrics
from panga.classify import ReadRules

__all__ = ('BlockingAnalysis',)

logger = logging.getLogger(__name__)


class BlockingAnalysis(StageBase):

    def  __init__(self, splitter, classifier, kwargs=None):
        """Join unproductive reads together and reanalyse"""
        # provide a default config, and overwrite any of these options with
        # options provided in second_stage_config (a yaml file).
        conf = {'with_events':       True,
                'with_raw':          False,
                'with_states':       True,
                'window_bounds':     (0,1,3,500,900),
                'non_block_classes': ('pore', 'tba', 'strand', 'multiple', 'saturated', 'off'),
                'recovered_classes': ('strand', 'pore'),
                'state_classes':     ('strand', 'pore'),
                'to_inject':         (('median_current', -1, '_before', 0),
                                    ('median_current', 1, '_after', 0),
                                    ('initial_classification', -1, '_before', 'none'),
                                    ('initial_classification', 1, '_after', 'none')
                                    ),
                'time_since':        ('unproductive', 'strand', 'pore', 'bermuda'),
                'rules': [ 'bermuda = (duration,gt,900) & (median_current_1_to_3_s,gt,40) & (median_current_1_to_3_s,lt,150) & (median_current_500_to_900_s - median_current_1_to_3_s,gt,-90) & (median_current_500_to_900_s - median_current_1_to_3_s,lt,-25)',
                           'unproductive = (initial_classification, eq, "unproductive")',
                           'pore = (initial_classification, eq, "pore")',
                           'strand = (initial_classification, eq, "strand")',
                           'tba = (initial_classification, eq, "tba")',
                           'adapter = (initial_classification, eq, "adapter")',
                           'multiple = (initial_classification, eq, "multiple")'
                ],
        }
        if kwargs is not None:
            conf.update(kwargs)

        super(BlockingAnalysis, self).__init__(splitter, classifier)
        # get channel and fast5 from the splitter
        self.splitter = BlockingJoin(splitter.channel, fast5=splitter.fast5, with_events=conf['with_events'],
                                      with_raw=conf['with_raw'], with_states=conf['with_states'],
                                      non_block_classes=conf['non_block_classes'])
        self.metrifier = BlockingMetrics(self.splitter, window_bounds=conf['window_bounds'])
        self.classifier = ReadRules(self.metrifier, conf['rules'],
                                    to_inject=conf['to_inject'], time_since=conf['time_since'],
                                    recovered_classes=conf['recovered_classes'], state_classes=conf['state_classes'],
        )

    def _process_reads(self, reads, metrics):
        # construct a new pipeline
        # make a new generator of read objects
        # tee because we want to yield reads and metrics
        joined_reads_0, joined_reads_1 = itertools.tee(self.splitter._process_reads(reads, metrics))
        # make a new generator of blocking metrics
        block_metrics = self.metrifier.process_reads(joined_reads_0)
        # make a new generator of classified metrics
        classed_metrics = self.classifier.process_reads(block_metrics)
        return joined_reads_1, classed_metrics


class BlockingJoin(Fast5Split):
    def __init__(self, channel, fast5=None, with_events=True, with_raw=False, with_states=True,
                 non_block_classes=('pore', 'tba', 'strand', 'multiple', 'saturated', 'off')):
        """Join togther reads which don't belong to non_block_classes.

        :param channel: channel for which to provide reads.
        :param fast5: fast5 file, only needed if with_events, with_raw or with_states is True.
        :param with_events: bool, whether to provide event data from the fast5
        :param with_raw: bool, whether to provide raw data from the fast5.
        :param with_states: if True, add channel state data (mux and channel states) from the fast5.
        :param non_block_classes: iterable of str, classes not to join together into unproductive periods.

        """
        self.with_states = with_states
        self.fast5 = fast5
        if (with_events or with_raw or self.with_states) and self.fast5 is None:
            raise RuntimeError('Raw, events or channel_states requested, but no fast5 provided')
        super(BlockingJoin, self,).__init__(fast5, channel, with_events=with_events, with_raw=with_raw)
        self.non_block_classes = non_block_classes

    @property
    def requires_meta(self):
        return (
            'start_time',
            'duration',
            'channel',
            'mux',
            'class',
        )

    @property
    def meta_keys(self):
        return (
            'start_time', 'duration', 'start_event', 'end_event',
            'mux', 'states', 'channel', 'mux_changes', 'mux_changes', 'bias_voltage_changes', 'initial_classification', 'initial_n_reads'
        )

    def _create_read_obj(self, reads, f5):
        read_meta = {}
        read_meta['start_time'] = reads[0]['start_time']
        read_meta['duration'] = reads[-1]['start_time'] + reads[-1]['duration'] - reads[0]['start_time']
        read_meta['start_event'] = reads[0]['start_event']
        read_meta['end_event'] = reads[-1]['end_event']
        read_meta['initial_n_reads'] = len(reads)
        read_meta['initial_classification'] = reads[0]['class']
        if read_meta['initial_classification'] not in self.non_block_classes:
            read_meta['initial_classification'] = 'unproductive'

        read_events = None
        if self.with_events:
            read_events = f5.get_events(self.channel, event_indices=(read_meta['start_event'], read_meta['end_event']))
            read_events = self._convert_event_fields(read_events, f5.sample_rate)
        read_raw = None
        if self.with_raw:
            read_raw = f5.get_raw(self.channel, times=(read_meta['start_time'], read_meta['start_time'] + read_meta['duration']), use_scaling=False)
        if self.with_states:  # add mux, channel states from fast5
            self._add_channel_states(f5, read_meta)
        return Read(events=read_events, raw=read_raw, meta=read_meta, channel_meta=self.channel_meta,
                           context_meta=self.context_meta, tracking_meta=self.tracking_meta)


    def _process_reads(self, reads, metrics):

        if self.with_events or self.with_raw or self.with_states:
            f5 = BulkFast5(self.fast5)
            # load channel, tracking and context meta so we don't need fast5
            # later to e.g. write fast5 files.
            self.load_fast5_meta(f5)

        reads_queue = []
        for read, meta in itertools.izip(reads, metrics):
            # TODO: at the moment we don't use the read objects as its simpler
            # just to load what we want from the fast5 but we could imagine
            # combining the events, raw, state_changes and mux_changes from
            # individual reads into a new read object, so we don't need a fast5
            logger.debug('Read: channel {} mux {} time {} class {}'.format(meta['channel'], meta['mux'], meta['start_time'], meta['class']))
            # if we have have accumulated any reads, yield them if the
            # well_id has changed. Note that unblock_voltage_1 and
            # common_voltage_1 both enumerate to 1, so this will not stop us
            # joining up blocks interrupted by flicks.
            if len(reads_queue) > 0 and not meta['mux'] == reads_queue[-1]['mux']:
                logger.debug('Detected change in mux, yielding existing reads')
                yield self._create_read_obj(reads_queue, f5)
                reads_queue = []  # prepare for next grouping of reads.
            # if the new read is not a block, if we have accumulated reads,
            # yield them, then yield this read
            if meta['class'] in self.non_block_classes:
                if len(reads_queue) > 0:
                    logger.debug('We have a non-block class, yielding existing reads')
                    yield self._create_read_obj(reads_queue, f5)
                    reads_queue = []  # prepare for next grouping of reads.
                logger.debug('We have a non-block class, yielding single read')
                yield self._create_read_obj([meta], f5)
            else:  # this is a block, so append to reads
                reads_queue.append(meta)
        if len(reads_queue) > 0:  # if we reach the end of the run, yield the block
            yield self._create_read_obj(reads_queue, f5)

        if self.with_events or self.with_raw or self.with_states:
            f5.close()

