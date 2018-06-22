import numpy as np
import os
import sys
import tempfile

from collections import Counter, defaultdict
from panga.accumulate.base import AccumulateBase
from panga.util import add_prefix

__all__ = ('MetricSummary', 'CountClasses', 'ClassicMetricSummary', 'FilteredMetricSummary', 'ChannelReport')


class MetricSummary(AccumulateBase):
    def __init__(self, outpath=None, summary_file=None, class_filter=None, prefix='', meta=None):
        """Accumulate read metrics into a file of tab-separated values.

        :param outpath: output path for file.
        :param summary_file: filename for output, if `None` output is written
            to stdout.
        :param class_filter: restrict to accumulating defined classes of read.
        :param prefix: run identifier to include in summary file.
        :param meta: dict of meta data to output for each read.
                     keys are column names, values the value to output for each row.

        ..note:: if the output file pre-exists an attempt is made to append
            data to it (without writing a new header). The obviously cannot be
            achieved when output is to stdout.
        """
        super(MetricSummary, self).__init__(class_filter=class_filter)
        self.outpath = outpath
        self.filename = summary_file
        self.prefix = prefix
        self.columns = None
        self.meta = meta

        if self.filename is not None:
            self.filename = add_prefix(self.filename, self.prefix)
        self._open()
        self.converter = lambda x: str(int(x)) if isinstance(x, bool) else str(x)  # convert True/False to 1/0

    def _process_read(self, read_metrics):
        if self.columns is None:
            self.columns = read_metrics.keys()
            row = '\t'.join(self.columns)
            if self.meta is not None:
                row = '\t'.join([row,'\t'.join(self.meta.keys())])
            self.fh.write('{}\n'.format(row))

        row = '\t'.join(self.converter(read_metrics[x]) for x in self.columns)
        if self.meta is not None:
            row = '\t'.join([row,'\t'.join(map(self.converter, self.meta.values()))])
        self.fh.write('{}\n'.format(row))

    def _open(self):
        # Ensure the internal filehandle is open
        if self.filename is not None:
            self.fh = open(os.path.join(self.outpath, self.filename), 'a', 0)
        else:
            self.fh = sys.stdout

    def _close(self):
        # Close the internal filehandle
        if self.fh is not sys.stdout:
            self.fh.close()


class ClassicMetricSummary(AccumulateBase):
    def __init__(self, outpath=None, summary_file=None, prefix='', meta=None):
        super(ClassicMetricSummary, self).__init__(class_filter=None)  # process all read classes
        self.read_summary = MetricSummary(outpath, summary_file=summary_file, class_filter=None, prefix=prefix, meta=meta)
        fname, ext = os.path.splitext(summary_file)
        strand_summary_fname = '{}_strand{}'.format(fname, ext)
        self.strand_summary = MetricSummary(outpath, summary_file=strand_summary_fname, class_filter='strand', prefix=prefix, meta=meta)

    def _process_read(self, read_metrics):
        self.read_summary.process_read(read_metrics)
        self.strand_summary.process_read(read_metrics)


class FilteredMetricSummary(MetricSummary):
    def __init__(self, outpath=None, filtered_summary='read_summary_filtered.txt',
                 filter_classes=('strand', 'pore', 'adapter'),
                 filter_counts=None, filter_durations=None,
                 filter_sum_duration=None, prefix='', meta=None, filter_ch_mux=False,
                 ):
        """Generate a summary containing  only filtered_classes reads.
        :param outpath: output path for file.
        :param filtered_summary: str, filename for output.
        :param filter_classes: tuple of read classes to include in summary.
        :param filter_counts: dict specifying minimum number of reads per read
               class below which the chan nel/mux combination will not app ear in the
               filtered summary (e.g. {'strand': 10} to exclude channels with <10 strands.
        :param filter_durations: dict specifying minimum summed duration of reads per
               class below which the channel/mux combination will not app ear in the
               filtered summary (e.g. {'pore': 120} to exclude channels with <120 seconds of pore time.
        :param filter_sum_duration: int, sum of read durations below which a
               channel/mux combination will not appear in the filtered summary.
        :param prefix: run identifier to include in summary file.
        :param meta: dict of meta data to output for each read.
                     keys are column names, values the value to output for each row.
        :param filter_ch_mux: bool, whether to remove channel-mux combinations
               which ever were is_saturated, is_multiple or had class='multiple'

        """
        self.filter_ch_mux = filter_ch_mux
        self.filter_classes = filter_classes
        self.min_counts = filter_counts
        self.min_durations = filter_durations
        self.min_sum_duration = filter_sum_duration

        class_filter = filter_classes
        if filter_classes is not None:
            if (self.filter_ch_mux or
                self.min_sum_duration is not None or
                (self.min_counts is not None and not set(self.min_counts.keys()).issubset(filter_classes)) or
                (self.min_durations is not None and not set(self.min_durations.keys()).issubset(filter_classes))):
                # we will have to process all reads
                    class_filter = None

        # store count per channel per mux per read class
        self.class_durations = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # store count per channel per mux per read class
        self.class_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # store record of bad channels dict with keys [channel][mux]
        # (assumes a channel is good in the absense of saturation/multiple
        self.electrically_fails = defaultdict(lambda: defaultdict(bool))

        self.nreads = 0

        super(FilteredMetricSummary, self).__init__(outpath,
                                                    summary_file=filtered_summary,
                                                    class_filter=class_filter,
                                                    prefix=prefix, meta=meta)

    def _process_read(self, read_metrics):

        ch = read_metrics['channel']
        mux = read_metrics['mux']
        klass = read_metrics['class']

        self.class_counts[ch][mux][klass] += 1
        self.class_durations[ch][mux][klass] += read_metrics['duration']

        if self.filter_ch_mux and (read_metrics['is_saturated']
                                   or read_metrics['is_multiple']
                                   or klass == 'multiple'):
            self.electrically_fails[ch][mux] = True

        # skip reads we already know come from bad channel mux combinations
        # and apply our class filters
        if ( not self.electrically_fails[ch][mux]
             and (self.filter_classes is None or klass in self.filter_classes)
            ):
            super(FilteredMetricSummary, self)._process_read(read_metrics)
            self.nreads += 1


    def finalize(self):
        """If required, filter out reads from ch/mux combinations which were multiple/saturated at any point,
        or had less than the minimum count of each read class in self.min_counts.
        """
        self.fh.close()

        # store record of which muxes we want to output
        # will the be product of all filters
        get_true = lambda: True
        self.keep_ch_mux = defaultdict(lambda: defaultdict(get_true))
        dont_need_to_filter = True

        for ch, ch_data in self.class_durations.items():
            for mux, mux_durations in ch_data.items():
                keep = True
                if self.filter_ch_mux and self.electrically_fails[ch][mux]:
                    keep = False
                    #print 'Turning off ch {} mux {} due to electrical failure'.format(ch, mux)
                # set self.working_pore True for each chan/mux with more than the
                # specified summed duration per read class, so these chan/muxes do get outputed
                if keep and self.min_durations is not None:
                    for klass in self.min_durations:
                        if mux_durations[klass] < self.min_durations[klass]:
                            keep = False
                            #print 'Turning off ch {} mux {} due to min_durations'.format(ch, mux)
                # set self.working_pore True for each chan/mux with more than the
                # specified number of reads, so these chan/muxes do get outputed
                if keep and self.min_counts is not None:
                    mux_counts = self.class_counts[ch][mux]
                    for klass in self.min_counts:
                        if mux_counts[klass] < self.min_counts[klass]:
                            keep = False
                            #print 'Turning off ch {} mux {} due to min_counts'.format(ch, mux)
                if keep and self.min_sum_duration is not None:
                    if np.sum(mux_durations.values()) < self.min_sum_duration:
                        keep = False
                        #print 'Turning off ch {} mux {} due to min_sum_durations'.format(ch, mux)

                self.keep_ch_mux[ch][mux] = keep
                dont_need_to_filter *= keep

        if self.nreads > 0 and not dont_need_to_filter:

            # loop over summary file and write out reads we want to keep to a temporary file, before overwriting
            # the summary file with the tmp file. Avoid loading entire summary into memory, they can be several GB.
            source_fh = open(self.fh.name, 'r')
            header = source_fh.readline()
            cols = header.strip().split('\t')
            if self.meta is not None:
                expected_cols = self.columns + self.meta.keys()
            else:
                expected_cols = self.columns
            if len(cols) != len(expected_cols) or not all(str(c[0]) == str(c[1]) for c in zip(cols, expected_cols)):
                raise RuntimeError('The filtered summary file has changed since it was written. Expected {}, found {}\n'.format(expected_cols, cols))
            ch_col = cols.index('channel')
            mux_col = cols.index('mux')
            tmp_f = tempfile.NamedTemporaryFile(prefix='tmp', dir=self.outpath, delete=True)
            tmp_fname = tmp_f.name
            tmp_f.close()  # if we were not to use this, the file would be only accessible by the user who created it

            with open(tmp_fname, 'w') as tmp_fh:
                tmp_fh.write(header)
                for line in source_fh:
                    split_line = line.split('\t')
                    ch = int(split_line[ch_col])
                    mux = int(split_line[mux_col])
                    if self.keep_ch_mux[ch][mux]:
                        tmp_fh.write(line)
            source_fh.close()
            os.rename(tmp_fname, self.fh.name)


class CountClasses(AccumulateBase):
    def __init__(self):
        """Accumulate read metrics by counting the classes of observed reads.
        On finalizing, prints counter to stdout.
        """
        super(CountClasses, self).__init__()
        self.counter = Counter()

    def _process_read(self, read_metrics):
        self.counter[read_metrics['class']] += 1

    def finalize(self):
        print self.counter


class ChannelReport(AccumulateBase):
    def __init__(self, outpath=None, channel_report='channel_report.txt',
                 report_class='saturated', end_of_read=True, prefix='',
                 ):
        """Report for each channel/mux whether the channel contains a report_class read and when the first such read occured.
        :param outpath: output path for file.
        :param channel_report: str, filename for output.
        :param report_class: str, class of reads to report on.
        :param end_of_read: bool, if true, report the end time of the first matching read, else report the start.

        """
        super(ChannelReport, self).__init__(class_filter=report_class)  # only process report_class reads
        self.outpath = outpath
        self.filename = channel_report
        self.report_class = report_class
        self.end_of_read = end_of_read
        self.prefix = prefix

        # store whether a channel contains the report_class class per channel per mux
        self.has_read_class = defaultdict(lambda: defaultdict(bool))
        # store time when the first report_class read occured
        self.first_time = defaultdict(lambda: defaultdict(float))

    def _process_read(self, read_metrics):

        ch = read_metrics['channel']
        mux = read_metrics['mux']
        klass = read_metrics['class']

        # only do something if this is the first time we have seen a
        # report_class read
        if not self.has_read_class[ch][mux]:
            self.has_read_class[ch][mux] = True
            self.first_time[ch][mux] = read_metrics['start_time']
            if self.end_of_read:
                self.first_time[ch][mux] += read_metrics['duration']

    def finalize(self):
        """Write the channel report.
        """
        filepath = add_prefix(self.filename, self.prefix)
        if self.outpath is not None:
            filepath = os.path.join(self.outpath, filepath)

        with open(filepath, 'w') as fh:
            row = '\t'.join(['channel', 'mux', 'first_{}_time'.format(self.report_class)])
            fh.write('{}\n'.format(row))
            for ch, muxes in self.has_read_class.items():
                for mux in muxes:
                    row = '\t'.join(map(str, [ch, mux, self.first_time[ch][mux]]))
                    fh.write('{}\n'.format(row))
