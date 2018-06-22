import numpy as np
import os
import shutil
import tempfile
import unittest
from panga import read_builder
from panga.util import add_prefix


class RegressionTest(unittest.TestCase):
    """Regression test Panga read_builder against Ossetra strand_builder
    using a rules config file."""


    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def get_ref_filename(cls):
        return 'example_strand_summary.txt'

    @classmethod
    def get_read_builder_args(cls):
        # run panga read_builder using strand_builder rules
        fast5 = os.path.join(cls.data_path, 'example_reads.fast5')
        config = os.path.join(cls.data_path, '3.6kb_strand_no_read_class.yml')

        args = ['--fast5', fast5,
                '--config', config,
                '--channels', '1,2',
                '--outpath', cls.out_path,
                '--summary_file', 'panga_read_summary.tsv',
                '--prefix', 'test',
                '--meta', '{"run":"example"}',
        ]
        return args

    @classmethod
    def setUpClass(cls):

        cls.data_path = os.path.join(os.path.dirname(__file__), 'data')
        cls.out_path = tempfile.mktemp(dir=cls.data_path)
        args = cls.get_read_builder_args()  # a list
        read_builder.main(args=args)
        args = read_builder.get_argparser().parse_args(args)  # an argstore obj
        # load in the read_builder strand summary
        cls.rb = np.genfromtxt(os.path.join(args.outpath, add_prefix(args.summary_file, args.prefix)), names=True, dtype=None)
        cls.sb = np.genfromtxt(os.path.join(cls.data_path, cls.get_ref_filename()), names=True, dtype=None)
        # remove reads with mux not in [1,2,3,4] as differences in mux
        # manipulation between panga and ossetra are expected to lead to
        # differences in the number of reads
        cls.sb = cls.sb[np.logical_and(cls.sb['mux'] > 0, cls.sb['mux'] < 5)]
        cls.rb = cls.rb[np.logical_and(cls.rb['mux'] > 0, cls.rb['mux'] < 5)]

        # define column name mapping for those numeric columns which are common
        cls.rb_to_sb_col_map = {
            'duration': 'strand_duration',
            'median_current_after': 'pore_after',
            'channel': 'channel',
            'drift': 'drift',
            'end_event': 'end_event',
            'median_current': 'median_current',
            'median_dwell': 'median_dwell',
            'median_sd': 'median_sd',
            'mux': 'mux',
            'num_events': 'num_events',
            'range_current': 'range_current',
            'start_event': 'start_event',
            'start_time': 'start_time',
        }

        print '\n* {}'.format(str(cls))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_path)

    def test_000_n_reads(self):
        self.assertEqual(len(self.rb), len(self.sb))

    def test_001_values(self):
        for rb_col, sb_col in self.rb_to_sb_col_map.items():
            try:
                assert np.allclose(self.rb[rb_col], self.sb[sb_col], atol=0.001)
            except AssertionError as error:
                raise AssertionError('Col {} differed'.format(rb_col))

    def test_002_classes(self):
        # first and last reads for each channel have class 'unclassed' in
        # read_builder, but not in strand_builder
        for chan in np.unique(self.sb['channel']):
            mask = self.sb['channel'] == chan
            assert np.all(self.rb[mask]['class'][1:-1] == self.sb[mask]['read_class'][1:-1])

    def test_003_median_current_before(self):
        # first reads for each channel have pore_before from mux 0 in
        # strand_builder but in read_builder mux 0 reads are excluded and
        # median_current_before is set to zero.
        for chan in np.unique(self.sb['channel']):
            mask = self.sb['channel'] == chan
            assert np.allclose(self.rb[mask]['median_current_before'][1:], self.sb[mask]['pore_before'][1:], atol=0.001)

    def test_004_run(self):
        # this is a string, so check for it be equal
        print 'rb {} sb {}'.format(np.unique(self.rb['run']), np.unique(self.sb['run']))
        np.testing.assert_equal(self.sb['run'], self.rb['run'])


class MinknowClassRegressionTest(RegressionTest):
    """Class to regression test Panga read_builder against Ossetra strand_builder
    using read classes taken directly from the fast5 file."""

    @classmethod
    def get_ref_filename(cls):
        return 'minknow_class_test_strand_summary.txt'

    @classmethod
    def get_read_builder_args(cls):

        fast5 = os.path.join(cls.data_path, 'example_reads.fast5')
        config = os.path.join(cls.data_path, 'minknow_classes.yml')
        args = ['--fast5', fast5,
                '--config', config,
                '--channels', '1,2',
                '--outpath', cls.out_path,
                '--summary_file', 'panga_read_summary.tsv',
                '--prefix', 'metric_class',
                '--meta', '{"run":"test"}',
        ]
        return args

    def test_002_classes(self):
        # all classes should be the same.
        assert np.all(self.rb['class'] == self.sb['read_class'])


if __name__ == "__main__":
    unittest.main()
