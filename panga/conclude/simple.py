import os
import logging

from fast5_research.fast5 import Fast5

from panga.conclude import ConcludeBase
from panga.util import ensure_dir_exists
from panga.util import add_prefix

__all__ = ('Fast5Write',)

logger = logging.getLogger(__name__)


class Fast5Write(ConcludeBase):

    def  __init__(self, channel, outpath, prefix, fast5_class_filter=None):
        """Write reads to fast5 files with meta copied from a reference bulk file.

        :param fast5: fast5 file from which to copy meta data.
        """
        super(Fast5Write, self).__init__(class_filter=fast5_class_filter)
        self.channel = channel
        self.outpath = os.path.join(outpath, 'reads')
        self.prefix = prefix

        ensure_dir_exists(self.outpath)

        # initialise a read number which will be incremented
        self.n_reads = 0


    def _process_read(self, read, read_metrics):
        self.n_reads += 1

        filename = 'read_ch{}_file{}.fast5'.format(self.channel, self.n_reads)
        filename = add_prefix(filename, self.prefix)
        # add filename to read_metrics so it can be reported in summaries
        read_metrics['filename'] = filename
        filename = os.path.join(self.outpath, filename)

        channel_id = {
            'channel_number': self.channel,
            'range': read.channel_meta['range'],
            'digitisation': read.channel_meta['digitisation'],
            'offset': read.channel_meta['offset'],
            'sample_rate': read.channel_meta['sample_rate'],
            'sampling_rate': read.channel_meta['sample_rate']
        }
        if read.events is None:
            raise RuntimeError('Read has no events data, cannot write fast5')
        events = read.events
        read_id = {
            'start_time': events['start'][0],
            'duration': events['start'][-1] + events['length'][-1] - events['start'][0],
            'read_number': self.n_reads,
            'start_mux': read_metrics['mux'],
            'read_id': read.meta['read_id'],
            'scaling_used': 1,
            'median_before': read_metrics['median_current_before'],
        }

        with Fast5.New(filename, 'a', tracking_id=read.tracking_meta,
                       context_tags=read.context_meta, channel_id=channel_id) as h:
            h.set_read(events, read_id)
            if read.raw is not None:
                h.set_raw(read.adc_raw)


    def process_read(self, read, read_metrics):
        """Add a key:value pair 'filename':'none' to read_metrics before calling base class process_read"""

        # set filename for every read as required for consistent summary output.
        read_metrics['filename'] = 'none'
        super(Fast5Write, self).process_read(read, read_metrics)
