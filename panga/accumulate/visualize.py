from copy import deepcopy
from collections import defaultdict, OrderedDict
from itertools import chain, cycle, islice
import numpy as np
import os
from math import ceil
import matplotlib; matplotlib.use('Agg', warn=False)  # enforce non-interactive backend
from matplotlib import pyplot as plt

from panga.accumulate.base import AccumulateBase
from panga.util import add_prefix

__all__ = ('DutyTimePlot', 'DutyTimeDistPlot', 'EventRatePlot',)


class DutyTimePlot(AccumulateBase):

    def __init__(self, outpath, prefix, time_binning=2, class_filter=None, outfile='duty_time'):
        """Plot stacked bar charts of channel states accumulated by time.

        :param outpath: output path for plot.
        :param prefix: prefix (prefixed to output files)
        :param time_binning: binning for time (in minutes).
        :param class_filter: restrict analysis to this class of read
        :param outfile: str, name of generated plot and txt files (without extension)
        """
        super(DutyTimePlot, self).__init__(class_filter=class_filter)
        self.outpath = outpath
        self.outfile = outfile
        self.prefix = prefix
        self.time_binning = time_binning * 60
        self.bin_width_hours = self.time_binning / 3600.0
        self.last_bin = 0  # in units of time_binning
        self.class_filter = class_filter
        # store accumulated counts as dict with keys [channel][mux][time_bin][class]
        self.accumulators = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
        self.observed_classes = set()
        # to plot counts of well changes, need to keep track of what
        # last well was for each channel
        self.well_muxes = set([1,2,3,4])
        self.prev_well = defaultdict(lambda: defaultdict(int))  # last well for each channel
        # well changes in each time bin
        self.well_changes = defaultdict(int)
        # TODO: pull this out into config
        self.static_classes = OrderedDict([
            ('strand','LawnGreen'),
            ('strand1','LimeGreen'),
            ('bound_dna','LightGreen'),
            ('unbound_dna','Olive'),
            ('pore','Green'),
            ('unavailable','DarkRed'),
            ('long_block','SaddleBrown'),
            ('zero','Indigo'),
            ('off','Red'),
            ('multiple','MediumVioletRed'),
            ('saturated','Black'),
            ('unclassed','Gray'),
        ])


    def _process_read(self, read_metrics):
        """add read to duty time analysis
        """
        klass = read_metrics['class']
        ch = read_metrics['channel']
        mux = read_metrics['mux']
        bins = self._split_span(read_metrics['start_time'], read_metrics['duration'])
        for time_bin, duration in bins.items():
            self.accumulators[ch][mux][time_bin][klass] += duration

        # if mux has changed to a new well, increment well change count and
        # update previous well
        if mux in self.well_muxes and mux != self.prev_well[ch]:
            self.well_changes[bins.keys()[0]] += 1
            self.prev_well[ch] = mux

        # update state data
        self.observed_classes.update({klass,})
        self.last_bin = max(self.last_bin, max(bins.keys()))


    def _split_span(self, start, duration):
        first_bin = int(start // self.time_binning)
        last_bin = int((start+duration) // self.time_binning)
        bins = dict()
        if first_bin == last_bin:
            # read wholely contained in one bin
            bins = {first_bin:duration}
        else:
            bins[first_bin] = (self.time_binning * (first_bin + 1)) - start
            bins[last_bin] = start + duration - (self.time_binning * last_bin)
            # intervening get full contribution
            bins.update({x:self.time_binning for x in xrange(first_bin+1, last_bin)})
        return bins

    def __group_bins(self):
        """Combine neighbouring bins for longer runs to avoid overcrowding duty time plot"""
        # for runs of two hours or less, keep bin size as it is.
        # but for runs twice as long, double bin size, etc..
        run_length_hours = int(round((self.last_bin + 1) * self.bin_width_hours))
        bins_to_join = run_length_hours // 2
        if bins_to_join > 1:
            self.last_bin /= bins_to_join
            self.bin_width_hours *= bins_to_join
            self.time_binning *= bins_to_join
            new_accumulator = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

            for ch, muxes in self.accumulators.items():
                for mux, bin_dict in muxes.items():
                    for time_bin, bin_duration in bin_dict.items():
                        new_time_bin = time_bin // bins_to_join
                        for klass in self.all_classes.keys():
                            new_accumulator[ch][mux][new_time_bin][klass] += bin_duration[klass]

            self.accumulators = new_accumulator


    def finalize(self):

        # rotate dict-of-dicts into array
        self.all_classes = deepcopy(self.static_classes)
        for klass in self.observed_classes:
            if klass not in self.all_classes:
                self.all_classes.update({
                    klass:choose_colour(klass)
                })
        self.__group_bins()  # adapt bin size to run length


        # do plotting
        plot_items = OrderedDict()
        channel_cols, channel_avail = self.__calc_channel_avail(txt_path=self.outfile + '_channel.txt')  # column names, data
        plot_items['channel activity'] = channel_cols, channel_avail
        # calculate inactive and pore availability by filtering out columns
        inactive_classes = ['multiple', 'saturated','off']
        plot_items['pore activity'] = self.__filter_channel_avail(channel_cols, channel_avail, exclude_classes=inactive_classes, renorm=True)
        plot_items['channel inactivity'] = self.__filter_channel_avail(channel_cols, channel_avail, include_classes=inactive_classes)
        # legacy availability calculated separately
        plot_items['legacy pore activity'] = self.__calc_channel_avail(txt_path=self.outfile + '_legacy.txt', legacy=True)
        plot_items['well changes'] = self.__get_well_changes()
        nrows = len(plot_items)
        fig, axes = plt.subplots(nrows=nrows, figsize=(8, 4 * nrows))
        fig.subplots_adjust(hspace=0.3)
        plot_path = os.path.join(self.outpath, add_prefix(self.outfile + '.png', self.prefix))
        for i, (title, (cols, data)) in enumerate(plot_items.items()):
            if title == 'well changes':
                DutyTimePlot.__plot_figure(axes[i], data, cols, self.bin_width_hours, {'well changes':'Black'}, title=title, ylabel='% Channels Changing Well')
            else:
                DutyTimePlot.__plot_figure(axes[i], data, cols, self.bin_width_hours, self.all_classes, title=title)
        fig.savefig(plot_path, bbox_inches='tight', dpi=200)

    def __get_well_changes(self):
        data = np.empty((self.last_bin + 1, 2))  # time, mux change count
        data[:,0] = [ i * self.bin_width_hours for i, _ in enumerate(data)]  # times
        data[:, 1] = [self.well_changes[i] for i in xrange(self.last_bin + 1)]
        # present as % channels with mux changes
        n_channels = len(self.accumulators)
        data[:, 1] *= (100.0 / n_channels)
        return ['time', 'well changes'], data

    def __filter_channel_avail(self, all_cols, channel_avail, include_classes=None, exclude_classes=None, renorm=False):
        filtered_cols = ['time']
        if include_classes is not None:
            filtered_cols.extend(include_classes)
        elif exclude_classes is not None:
            for col in all_cols:
                if col != 'time' and col not in exclude_classes:
                    filtered_cols.append(col)
        else:
            filtered_cols = all_cols

        # create filtered list in same order as all_cols
        col_inds = []
        for i, col in enumerate(all_cols):
            if col in filtered_cols:
                col_inds.append(i)

        filtered_cols = [ all_cols[i] for i in col_inds]
        avail = channel_avail[:, col_inds]
        if renorm:
            avail = deepcopy(avail)
            for row in avail:
                total = sum(row[1:])
                if total !=0:
                    row[1:] *= 100.0 / total
        return filtered_cols, avail

    def __calc_channel_avail(self, txt_path='duty_time_channel.txt', legacy=False):
        """Generate channel availability duty time data.

        This captures the availability and normalizes by all channels.

        :param: legacy, bool, if True, generate duty time data as strand_builder did.
                This dischards any channels which were not active at the start of the run.
        :return: (list of column names (str), array of duty time data)
        """
        data_per_ch_mux = []  # duty_time data for each mux
        n_classes = len(self.all_classes.keys())
        data_aggr = np.zeros((self.last_bin + 1, n_classes + 1))  # duty_time data summed over all muxes
        data_aggr[:,0] = [ i * self.bin_width_hours for i, _ in enumerate(data_aggr)]  # times

        for ch, muxes in self.accumulators.items():
            if legacy:
                #  discard any channels without a well mux - these are channels
                #  inactive throughout the run.
                if self.well_muxes.isdisjoint(muxes.keys()):
                    continue
            for mux, bin_dict in muxes.items():
                data = np.empty((self.last_bin + 1, 3 + n_classes))
                data[:, 0] = data_aggr[:,0]
                data[:, 1].fill(ch)
                data[:, 2].fill(mux)
                for i in xrange(self.last_bin + 1):
                    data[i, 3:] = [bin_dict[i][klass] for klass in self.all_classes.keys()]
                    data_aggr[i, 1:] += data[i, 3:]
                data_per_ch_mux.append(data)

        # Normalise total duration in each bin
        for data in chain([data_aggr], data_per_ch_mux):
            for i in range(self.last_bin + 1):
                bin_sum = np.sum(data_aggr[i, -n_classes:])
                if bin_sum != 0:
                    data[i, -n_classes:] *= (100.0 / bin_sum)

        # write per channel / mux data to text file
        cols = ['time', 'channel', 'mux'] + self.all_classes.keys()
        fmts = ['%.6f', '%i', '%i'] + n_classes * ['%.6f']
        txt_path = os.path.join(self.outpath, add_prefix(txt_path, self.prefix))
        np.savetxt(txt_path, np.concatenate(data_per_ch_mux), fmt=fmts, delimiter='\t', header='\t'.join(cols), comments='')

        cols_aggr = ['time'] + self.all_classes.keys()
        return cols_aggr, data_aggr


    @staticmethod
    def __plot_figure(axis, data, cols,  width_hours, all_classes, title="", ylabel='% Time Spent'):
        """Plot a figure to a png file.

        :param axis: axis to plot on.
        :param data: np.array. First col is time, other cols are percentages.
        :param cols: list of str, column names
        :param width_hours: bin width in hours
        :param all_classes: dict mapping each class to a plotting colour.
        """
        base = np.zeros(data.shape[0], dtype=float)
        p = []  # to store plot items
        times = data[:, 0]
        for i in range(1, len(cols)):
            p.append(axis.bar(times, data[:, i], bottom=base,
                            width=width_hours, color=all_classes[cols[i]],
                            align='edge' ))
            base += data[:, i]
        max_hour = int(ceil(np.max(times)))
        if max_hour < 1:
            max_hour = 1
        jump = int(max_hour / 24) + 1
        hours = range(0, max_hour + 1, jump)
        hour_labels = [str(h) for h in hours]
        # ax.xticks(hours, hour_labels)
        axis.set_xticks(hours)
        axis.set_xticklabels(hour_labels)
        axis.set_title('{} by Time'.format(title.title()))
        axis.set_ylabel(ylabel)
        axis.set_xlabel('Time (hours)')
        axis.set_xlim((0, max_hour))
        axis.set_ylim((0, 100))
        axis.legend(tuple([x[0] for x in p]), cols[1:], loc='upper left',
                   bbox_to_anchor=(1.0, 1.0), prop={'size':7}, frameon=False)


class DutyTimeDistPlot(AccumulateBase):

    def __init__(self, outpath, prefix, nbins=10, duty_class='strand', outfile='duty_time_dist'):
        """Plot bar chart histogram of % time a channel spends in a particular read class.

        :param outpath: str, output directory.
        :param prefix: prefix (prefixed to output files)
        :param nbins: number of percentage bins.
        :param duty_class: class for which to plot the duty time distribution
        :param outfile: str, name of generated plot and txt files (without extension)
        """
        self.outpath = outpath
        self.prefix = prefix
        self.nbins = nbins
        self.outfile = outfile
        # we can't call duty_class class_filter, otherwise
        # AccumulateBase.process_read will process only strand reads
        # but we need to process all reads to get total durations per channel
        self.duty_class = duty_class
        super(DutyTimeDistPlot, self).__init__(class_filter=None)  # process all reads

        self.class_accumulator = defaultdict(float)
        self.accumulator = defaultdict(float)

    def _process_read(self, read_metrics):
        ch = read_metrics['channel']
        dur = read_metrics['duration']
        self.accumulator[ch] += dur
        if read_metrics['class'] == self.duty_class:
            self.class_accumulator[ch] += dur

    def finalize(self):
        percentages = [ 100.0 * self.class_accumulator[ch] / self.accumulator[ch]
                        for ch in self.accumulator.keys() ]

        # plot data to a png file
        plot_path = os.path.join(self.outpath, add_prefix(self.outfile + '.png', self.prefix))
        DutyTimeDistPlot.__plot_figure(plot_path, percentages, self.nbins, self.duty_class)

        # write data to text file
        txt_path = os.path.join(self.outpath, add_prefix(self.outfile + '.txt', self.prefix))
        with open(txt_path, mode='a') as fh:
            fh.write('channel\tpercentage_time_{}\n'.format(self.duty_class))
            for ch, percentage in zip(self.accumulator.keys(), percentages):
                fh.write('{}\t{}\n'.format(ch, percentage))


    @staticmethod
    def __plot_figure(path, data, nbins, duty_class):
        """Plot bar chart histogram of % time a channel spends in a particular read class.

        :param path: output file path for plot.
        :param data: data from which to plot a histogram.
        :param nbins: number of percentage bins.
        :param duty_class: class to which data corresponds (used for labelling plot)
        """
        plt.figure(figsize=(4, 3))
        plt.hist(data, range=(0.0, 100.0), bins=nbins, zorder=3, edgecolor='none')
        plt.xlim((0.0, 100.0))
        plt.title('Duty Time Distribution')
        plt.xlabel('% Time {}'.format(duty_class.capitalize()))
        plt.ylabel('No. Pores')
        plt.grid(True, zorder=0)
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()


class EventRatePlot(AccumulateBase):

    def __init__(self, outpath, prefix, time_binning=60, class_filter='strand', outfile='event_rate.png'):
        """Plot bar chart of event rate (events per second) for the given class filter.

        :param outpath: output path for plot.
        :param prefix: prefix (prefixed to output files)
        :param time_binning: binning for time (in minutes).
        :param class_filter: restrict analysis to this class of read, if None process all reads.
        :param outfile: str, name of generated plot.

        """
        super(EventRatePlot, self).__init__(class_filter=class_filter)
        self.outpath = outpath
        self.prefix = prefix
        self.time_binning = time_binning * 60  # in seconds
        self.bin_width_hours = self.time_binning / 3600.0
        self.last_bin = 0  # in units of time_binning
        self.class_filter = class_filter
        self.outfile = outfile
        # store summed number of events per read and read durations dicts with keys[time_bin]
        self.sum_num_events = defaultdict(int)
        self.sum_duration = defaultdict(float)

    def _process_read(self, read_metrics):
        """add read to duty time analysis
        """
        klass = read_metrics['class']
        bins = self._split_span(read_metrics['start_time'], read_metrics['duration'])
        for time_bin, duration in bins.items():
            self.sum_num_events[time_bin] += read_metrics['num_events']
            self.sum_duration[time_bin] += duration

        # update state data
        self.last_bin = max(self.last_bin, max(bins.keys()))

    def _split_span(self, start, duration):
        first_bin = int(start // self.time_binning)
        last_bin = int((start+duration) // self.time_binning)
        bins = dict()
        if first_bin == last_bin:
            # read wholely contained in one bin
            bins = {first_bin:duration}
        else:
            bins[first_bin] = (self.time_binning * (first_bin + 1)) - start
            bins[last_bin] = start + duration - (self.time_binning * last_bin)
            # intervening get full contribution
            bins.update({x:self.time_binning for x in xrange(first_bin+1, last_bin)})
        return bins


    def finalize(self):

        data = np.zeros((self.last_bin + 1))  # average event rate in each time period
        times =  np.arange(0, (self.last_bin + 1) * self.bin_width_hours, self.bin_width_hours)

        # Calculate average duration of an event
        # invert  so we have a rate or speed (events per second)
        for i in xrange(self.last_bin + 1):
            if self.sum_duration[i] > 0:
                data[i] = self.sum_num_events[i] / self.sum_duration[i]

        # do plotting
        nrows = 1
        fig, axis = plt.subplots(nrows=nrows, figsize=(8, 4 * nrows))
        fig.subplots_adjust(hspace=0.3)
        plot_path = os.path.join(self.outpath, add_prefix(self.outfile, self.prefix))
        axis.bar(times, data, width=self.bin_width_hours, align='edge')
        max_hour = int(ceil(np.max(times)))
        if max_hour < 1:
            max_hour = 1
        jump = int(max_hour / 24) + 1
        hours = range(0, max_hour + 1, jump)
        hour_labels = [str(h) for h in hours]
        # ax.xticks(hours, hour_labels)
        axis.set_xticks(hours)
        axis.set_xticklabels(hour_labels)
        axis.grid()
        klass_filter_str = self.class_filter if self.class_filter is not None else ''
        axis.set_title('Average {} event rate by time'.format(klass_filter_str))
        axis.set_ylabel('Speed (events per second)')
        axis.set_xlabel('Time (hours)')
        fig.savefig(plot_path, bbox_inches='tight', dpi=200)


def random_colour(low=0.6, high=0.95):
    """Choose a random colour, return hex RGB string.

    :param low: restrict each item of RGB colour space to this lower bound.
    :param high: restrict colour space upper bound.
    """

    space = 256
    colours = np.random.uniform(low=low*space, high=high*space, size=3).astype(int)
    return '#{}{}{}'.format(*(hex(x)[2:4] for x in colours))


def choose_colour(seed, low=0.6, high=0.95):
    """Deterministically choose a colour from seed data, return hex RGB string.

    :param seed: seed data for choosing colour, multiple calls with same seed
        will return an identical result.
    :param low: restrict each item of RGB colour space to this lower bound.
    :param high: restrict colour space upper bound.
    """
    if isinstance(seed, int):
        # ints hash to their own value, which causes clashes in the below
        seed = str(seed)

    space = 256
    low *= space
    high *= space
    fill = [hex(int(0.5*(high + low)))[2:4].zfill(2)] * 3
    hexstring = hex(hash(seed))[3:]
    # try to maintain diversity whilst restricting range
    rgb = []
    rgb_rough = []
    for i in xrange(0, len(hexstring), 2):
        h = hexstring[i:i+2]
        hint = int(h, 16)
        if hint > low and hint < high:
            rgb.append(h)
        rgb_rough.append(h)
    return '#{}{}{}'.format(*list(islice(chain(cycle(rgb), rgb_rough, fill), 3)))

