import logging
import numpy as np
import os
import string

from collections import defaultdict, OrderedDict
from copy import deepcopy
from scipy.stats import expon
from panga.accumulate.base import AccumulateBase
from panga.accumulate.simple import FilteredMetricSummary
from panga.util import add_prefix
from panga.accumulate.visualize import choose_colour

import matplotlib; matplotlib.use('Agg', warn=False)  # enforce non-interactive backend
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


logger = logging.getLogger(__name__)


class BlockingReport(AccumulateBase):
    def __init__(self, outpath=None, filtered_summary='read_summary_filtered.txt',
                 filter_classes=None, filter_counts=None, filter_durations=None, filter_sum_duration=None,
                 prefix='', meta=None, filter_ch_mux=False,
                 block_class='bermuda', block_ch_report='block_channel_report.txt', block_run_report='block_run_report.txt',
                 static_classes = (('bermuda', 'Red'),
                                   ('unproductive', 'Black'),
                                   ('pore', 'Blue'),
                                   ('strand', 'LawnGreen'),
                                   ('adapter', 'Olive')),
                 make_plots=True,
                 plot_hist_cols=('time_to_first_bermuda',
                                 'sum_duration_pore_to_first_bermuda',
                                 'sum_duration_unproductive_to_first_bermuda',
                                 ),
                 plot_hist_sum_cols={'n_adapter_or_strand_to_first_bermuda':
                                         ('n_adapter_to_first_bermuda',
                                         'n_strand_to_first_bermuda'),
                                     'sum_duration_productive_to_first_bermuda':
                                         ('sum_duration_pore_to_first_bermuda',
                                         'sum_duration_strand_to_first_bermuda',
                                         'sum_duration_adapter_to_first_bermuda',)
                 },
                 hist_bins=20,
                 plot_count_cols=('n_bermuda',)
                 ):
        """Generate a FilteredSummary and perform some post-procesing of it,
        generating run-level and channel-level aggregated results.
        :param outpath: output path for file.
        :param filtered_summary: str, filename for output.
        :param filter_classes: tuple of read classes to include in summary.
        :param filter_counts: dict specifying minimum number of reads per read
               class below which the channel/mux combination will not app ear in the
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
        :param block_class: block class for reporting n_to_first etc.
        :param block_ch_report: str, filename of per-channel report
        :param block_run_report: str, filename of per-run report
        :param static_classes: iterable of tuples of str (class, colour) for stacked duty-time plot.
        :param make_plots: bool, whether to make plots.
        :param plot_hist_cols: iterable of str, columns for which to plot a histogram from pre-channel/mux results.
        :param plot_hist_sum_cols: dict, specifying which cols to sum before plotting a histogram from pre-channel/mux results.
               keys are names of the summed data, keys are iterables of columns names.
        :param hist_bins: int, number of histogram bins.
        :param plot_count_cols: iterable of str specifying which metrics for which to plot a histogram with bin_size 1 over ch/muxes.

        """
        self.filtered_summary = FilteredMetricSummary(outpath, filtered_summary=filtered_summary,
                                                      filter_classes=filter_classes,
                                                      filter_counts=filter_counts,
                                                      filter_durations=filter_durations,
                                                      filter_sum_duration=filter_sum_duration,
                                                      prefix=prefix,
                                                      meta=meta,
                                                      filter_ch_mux=filter_ch_mux)
        self.filtered_summary.class_filter = None  # force processing of every read
        super(BlockingReport, self).__init__(class_filter=self.filtered_summary.class_filter)

        self.block_class = block_class

        # for ch/muxes without a block, we don't want to put zeros for
        # n_reads to block, or sum_duration to_block, but rather make it
        # chear that for those channels, there is no block, and thus no value
        get_nan = lambda: np.nan  # default dict arg needs to be callable
        get_m1 = lambda: -1  # default dict arg needs to be callable, int can't be nan
        # record time to first block for each ch / mux
        self.time_to_block = defaultdict(lambda: defaultdict(get_nan))
        # record n reads of each class to first block for each ch / mux / klass
        self.n_reads_to_block = defaultdict(lambda: defaultdict(lambda: defaultdict(get_m1)))
        self.duration_reads_to_block = defaultdict(lambda: defaultdict(lambda: defaultdict(get_nan)))

        self.block_ch_report = block_ch_report
        self.block_run_report = block_run_report

        self.static_classes = OrderedDict(static_classes)
        self.make_plots = make_plots
        self.plot_hist_cols = plot_hist_cols
        self.plot_hist_sum_cols = plot_hist_sum_cols
        self.hist_bins = hist_bins
        self.plot_count_cols = plot_count_cols

    def _process_read(self, read_metrics):
        self.filtered_summary.process_read(read_metrics)
        ch = read_metrics['channel']
        mux = read_metrics['mux']
        klass = read_metrics['class']

        if klass == self.block_class and self.filtered_summary.class_counts[ch][mux][klass] == 1:
            # this is the first block for this ch/mux
            self.n_reads_to_block[ch][mux] = deepcopy(self.filtered_summary.class_counts[ch][mux])
            self.duration_reads_to_block[ch][mux] = deepcopy(self.filtered_summary.class_durations[ch][mux])
            self.time_to_block[ch][mux] = np.sum(self.duration_reads_to_block[ch][mux].values()) - read_metrics['duration']

    def finalize(self):
        """Finalize filtered_summary and perform post_processing.
        """
        self.filtered_summary.finalize()

        # get ch/muxes and classes from channels which have not been filtered out.
        self.klasses = set()
        # force self.klasses to include self.block_class so that even if we
        # don't have any blocks, we see e.g. pcnt_time_block 0.0 in the outputs
        self.klasses.update([self.block_class])

        for ch, ch_muxes in self.filtered_summary.class_counts.items():
            for mux in ch_muxes:
                if self.filtered_summary.keep_ch_mux[ch][mux]:
                    self.klasses.update(ch_muxes[mux].keys())

        # we don't want to report the number of blocks before the first block..
        self.to_first_klasses = self.klasses - set([self.block_class])

        ch_mux_results = self._make_channel_report()
        if len(ch_mux_results) > 0:
            ch_mux_results_ar, run_results = self._make_run_report(ch_mux_results)

            if self.make_plots:
                self.generate_plots(ch_mux_results_ar, run_results)


    def _make_channel_report(self):
        """create summary with one row per channel/mux combination"""
        durations = self.filtered_summary.class_durations
        counts = self.filtered_summary.class_counts

        ch_mux_results = []
        for ch, keep_muxes in self.filtered_summary.keep_ch_mux.items():
            for mux in keep_muxes:
                if keep_muxes[mux]:
                    d = OrderedDict()
                    d['channel'] =  ch
                    d['mux'] =  mux
                    d['time_to_first_{}'.format(self.block_class)] =  self.time_to_block[ch][mux]
                    d['n_{}'.format(self.block_class)] =  counts[ch][mux][self.block_class]
                    d['sum_durations'] =  np.sum(durations[ch][mux].values())
                    # add percentage times
                    d.update({ 'pcnt_time_{}'.format(klass):
                              100.0*durations[ch][mux][klass]/d['sum_durations'] for klass in self.klasses})
                    # add summed klass counts to first block
                    d.update({ 'n_{}_to_first_{}'.format(klass, self.block_class):
                              self.n_reads_to_block[ch][mux][klass] for klass in self.to_first_klasses})
                    # add summed klass durations to first block
                    d.update({ 'sum_duration_{}_to_first_{}'.format(klass, self.block_class):
                              self.duration_reads_to_block[ch][mux][klass] for klass in self.to_first_klasses})
                    ch_mux_results.append(d)

        ch_report_fp = os.path.join(self.filtered_summary.outpath,
                                    add_prefix(self.block_ch_report, prefix=self.filtered_summary.prefix))
        with open(ch_report_fp, 'w') as fh:
            if len(ch_mux_results) > 0:
                cols = ch_mux_results[0].keys()
                convert = self.filtered_summary.converter
                meta = {}
                if self.filtered_summary.meta is not None:
                    meta = self.filtered_summary.meta
                meta_vals = [convert(x) for x in meta.values() ]
                fh.write('{}\n'.format('\t'.join(cols + meta.keys())))
                [ fh.write('\t'.join([convert(d[x]) for x in cols] + meta_vals) + '\n') for d in ch_mux_results ]

        return ch_mux_results


    def _make_run_report(self, ch_mux_results):
        """create run report with all data aggregated into a single row"""

        #  convert per ch/mux data into a structured array so we can easily feed
        #  it into numpy to get medians / sums, etc.

        # ugly, assume any strings no longer than 20 char
        get_type = lambda x: type(x) if not isinstance(x, basestring) else 'S20'
        dtype = [ (str(key), get_type(value)) for key, value in ch_mux_results[0].items() ]
        results_ar = np.empty(len(ch_mux_results), dtype=dtype)
        for i, r in enumerate(ch_mux_results):
            results_ar[i] = tuple([ r[col] for col in results_ar.dtype.names ])

        agg = OrderedDict()
        agg['n_good_ch_mux'] = len(results_ar)
        agg['n_good_ch_mux_with_{}'.format(self.block_class)] = \
                len(np.where(results_ar['n_{}'.format(self.block_class)] > 0)[0])
        agg['sum_good_ch_mux_run_time'] = np.sum(results_ar['sum_durations'])

        col = 'time_to_first_{}'.format(self.block_class)
        # mask out any channels without a block
        mask = results_ar['n_{}'.format(self.block_class)] > 0
        agg['median_' + col] = np.median(results_ar[mask][col])
        # calculate mean of the exponential distribution to complement the median
        _, agg['exp_mean_' + col] = expon.fit(results_ar[mask][col], floc=0)


        for klass in self.klasses:
            # do percentage time in each class
            col = 'pcnt_time_{}'.format(klass)
            # weight % from each ch/mux by ch/mux sum duration, and renormalise
            agg[col] = np.sum(np.multiply(results_ar[col], results_ar['sum_durations'])) / agg['sum_good_ch_mux_run_time']

        for klass in self.to_first_klasses:
            # do median class count and duration before first block
            for col in ('n_{}_to_first_{}'.format(klass, self.block_class),
                        'sum_duration_{}_to_first_{}'.format(klass, self.block_class),
                        ):
                agg['median_' + col] = np.median(results_ar[mask][col])
                # calculate mean of the exponential distribution to complement the median
                _, agg['exp_mean_' + col] = expon.fit(results_ar[mask][col], floc=0)

        # do counts of channels with 0, 1, 2, 3, 4, 5 or >5 blocks
        col = 'n_{}'.format(self.block_class)
        bins = range(0,7) + [ max(7, np.max(results_ar[col]) + 1) ]
        counts, bins_np = np.histogram(results_ar[col], bins=bins, density=False)
        starts_counts = zip(bins_np, counts)
        for bin_start, count in starts_counts[:-1]:
            col = 'n_ch_mux_{}_{}'.format(bin_start,self.block_class)
            agg[col] = count
        rest_start, rest_count = starts_counts[-1]
        col = 'n_ch_mux_ge_{}_{}'.format(rest_start,self.block_class)
        agg[col] = rest_count


        run_report_fp = os.path.join(self.filtered_summary.outpath,
                                    add_prefix(self.block_run_report,
                                               prefix=self.filtered_summary.prefix))
        meta = self.filtered_summary.meta
        convert = self.filtered_summary.converter
        with open(run_report_fp, 'w') as fh:
            fh.write('{}\n'.format('\t'.join(agg.keys() + meta.keys())))
            fh.write('{}\n'.format('\t'.join([convert(v) for v in agg.values() + meta.values()])))

        return results_ar, agg


    def generate_plots(self, ch_results, run_results):
        outpath = self.filtered_summary.outpath
        prefix = self.filtered_summary.prefix
        for col in self.plot_hist_cols:
            mask = np.ones(len(ch_results), dtype=bool)
            if 'to_first_{}'.format(self.block_class) in col:
                # mask out channels without at least 1 block
                mask = ch_results['n_{}'.format(self.block_class)] > 0
            data = ch_results[col][mask]
            scale, units = self._get_scale_units(col)
            x_label = string.capwords(col.replace('_', ' ') + ' ' + units)
            data *= scale
            plot_path = os.path.join(outpath, add_prefix(col + '_hist.png', prefix))
            if len(data) == 0:  # our mask has filtered out all the data
                logger.info('Skipping plot {}, no data after masking.'.format(plot_path))
            else:
                self.plot_exp_hist(data, plot_path, x_label=x_label, prefix=prefix, n_bins=self.hist_bins)

        for sum_name, cols_to_sum in self.plot_hist_sum_cols.items():
            plot_path = os.path.join(outpath, add_prefix(sum_name + '_hist.png', prefix))
            mask = np.ones(len(ch_results), dtype=bool)
            if 'to_first_{}'.format(self.block_class) in sum_name:
                # mask out channels without at least 1 block
                mask = ch_results['n_{}'.format(self.block_class)] > 0
            rows = ch_results[mask]
            if len(rows) == 0:  # our mask has filtered out all the data
                logger.info('Skipping plot {}, no data after masking.'.format(plot_path))
            else:
                data = np.zeros(len(rows), dtype=rows[cols_to_sum[0]].dtype)
                for col in cols_to_sum:
                    data += rows[col]

                scale, units = self._get_scale_units(col)
                x_label = string.capwords(sum_name.replace('_', ' ') + ' ' + units)
                data *= scale
                self.plot_exp_hist(data, plot_path, x_label=x_label, prefix=prefix, n_bins=self.hist_bins)

        for col in self.plot_count_cols:
            plot_path = os.path.join(outpath, add_prefix(col + '_counts.png', prefix))
            scale, units = self._get_scale_units(col)
            x_label = string.capwords(col.replace('_', ' ') + ' ' + units)
            self.make_bar_chart(ch_results[col], plot_path, prefix=prefix, x_label=x_label)

        plot_path = os.path.join(outpath, add_prefix('pcnt_time.png', prefix))
        self.make_stacked_duty_time(run_results, plot_path, col_prefix='pcnt_time_',
                               colours_dict=self.static_classes,
                               x_label=prefix)

    @staticmethod
    def make_stacked_duty_time(results_dict, plot_path,
                               col_prefix='pcnt_time_',
                               colours_dict={},
                               x_label='', y_label='% Time',
                            ):
        klasses = []
        for col in results_dict:
            if col.startswith(col_prefix):
                klasses.append(col.replace(col_prefix, ''))
        # if we have been given a colours dict, use the colours and order of
        # columns there
        klass_colours  = { klass: colours_dict[klass] for klass in klasses }
        for klass in klasses:
            if klass not in klass_colours:
                klass_colours.update({
                    klass:choose_colour(klass)
                })
        base = 0.0
        p = []  # to store plot items
        fig, axis = plt.subplots(figsize=(1,4))
        for klass, colour in klass_colours.items():
            col = col_prefix + klass
            p.append(axis.bar(1,results_dict[col], bottom=base,
                              width=1, color=colour,
                              align='edge'))
            base += results_dict[col]
        names = [ string.capwords(klass) for klass in klass_colours.keys()]
        axis.legend(tuple([x[0] for x in p]), names, loc='upper left',
                    bbox_to_anchor=(-.3, 1.5), prop={'size':7}, frameon=False,
                    mode="expand", borderaxespad=0.3)
        axis.autoscale_view()
        axis.set_xticklabels([''])
        axis.set_xticks([])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xlim([0,3])
        axis.set_ylim([0,100])
        fig.savefig(plot_path, bbox_inches='tight', dpi=200)

    @staticmethod
    def plot_exp_hist(data, plot_path, prefix='', x_label='', n_bins=20):
        # do plotting
        nrows = 1
        fig, axis = plt.subplots(nrows=nrows, figsize=(5, 5 * nrows))
        fig.subplots_adjust(hspace=0.3)

        # if data type in int, we want to make sure the bins are no smaller than  1.
        bins = n_bins
        if isinstance(data[0], np.integer) or isinstance(data[0], int):
            max_val = np.max(data)
            if max_val < n_bins:
                bins = np.arange(0, max_val + 3, 1)
        axis.hist(data, bins=bins, normed=True)

        # fit expontial dist
        loc, scale = expon.fit(data, floc=0)
        x = np.linspace(0, np.max(data), 100)
        axis.plot(x,expon.pdf(x, loc=loc, scale=scale))
        axis.set_title('{} (exp fit mean {:5.1f})'.format(prefix, scale), size=10)
        axis.set_ylabel('Normalised Probability')
        axis.set_xlabel(x_label)
        fig.savefig(plot_path, bbox_inches='tight', dpi=200)

    @staticmethod
    def make_bar_chart(data, plot_path, prefix='', x_label='', bin_size=1):
        # do plotting
        nrows = 1
        fig, axis = plt.subplots(nrows=nrows, figsize=(5, 5 * nrows))
        fig.subplots_adjust(hspace=0.3)
        bins = np.arange(0, np.max(data) + 2, 1, dtype=int)
        axis.hist(data, bins=bins, normed=False)
        axis.set_title('{}'.format(prefix), size=10)
        axis.set_ylabel('Number of Channels')
        axis.set_xlabel(x_label)
        axis.set_xticks(bins)
        fig.savefig(plot_path, bbox_inches='tight', dpi=200)

    @staticmethod
    def _get_scale_units(col):
        units = ''
        scale = 1
        if 'n_' in col[:2]:
            units = ' (counts)'
            scale = 1
        elif 'duration' in col or 'time' in col:
            units = ' (mins)'
            scale = 1./60.0  # convert seconds to minutes
        return scale, units

