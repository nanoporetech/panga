import unittest

from panga.metric import MockMetrics
from panga.classify.read_rules import SubRule, Rule, ReadRules


class SubRulesTest(unittest.TestCase):
    """ A Subrule is one simple relation with the form:
           " (left_hand_side, operator, right_hand_side) "
    """
    def setUp(self):
        self.rules_text = [
                " (median_current, lt, 150) ",
                "(150, lt, median_current)",
                "(median_current + range_current , gt, 90)",
                "(40 + 50, gt, median_current + range_current)",

                "(pore_before, gt, 150)",
                "(150, lt, median_current$left)",
                "(read_class, eq, 'pore')",
                "(150 + 30, gt, pore_before)"
                ]
        self.requirements = [('median_current',),
                        ('median_current',),
                        ('median_current', 'range_current'),
                        ('median_current', 'range_current'),
                        ('median_current',),
                        ('median_current',),
                        ('read_class',),
                        ('median_current',),
                        ]
        self.rules = {text: SubRule(text) for text in self.rules_text}
        self.requirements = {text: req for text, req in
                                       zip(self.rules_text, self.requirements)}

    def test_000_requirements(self):
        """ Check that the subrules recognize which keys it needs to provide
        """
        for rule_text in self.rules_text:
            expected = self.requirements[rule_text]
            obtained = self.rules[rule_text].requires
            self.assertItemsEqual(expected, obtained)

    def test_001_evaluations(self):
        """ Check the result of the evaluation of different inputs
        """
        pore = {'read_class': 'pore', 'range_current': 5, 'median_current': 190}
        strand = {'read_class': "'user1'", 'range_current': 30, 'median_current': 70}

        # TODO: Possibly we would like to raise Error, because with a single
        # TODO: input in the metric, left_pore makes no sense!

        # Expected1 needs to follow the order of self.rules_text
        input1 = [{'median_current': 170, 'range_current': 60, 'read_class': 'pore'}]
        expected1 = [False, True, True, False, True, True, True, True]

        # Expected2 needs to follow the order of self.rules_text
        input2 = [pore, strand, pore]
        expected2 = [True, False, True, False, True, True, False, False]

        # Expected3 needs to follow the order of self.rules_text
        input3 = [strand, pore, strand]
        expected3 = [False, True, True, False, False, False, True, True]

        input_list = [input1, input2, input3]
        expected_list = [expected1, expected2, expected3]
        zip_list = zip(input_list, expected_list)
        for ii, (inputs, expected) in enumerate(zip_list):
            # Check that we know the expected value foe every rule
            assert len(expected) == len(self.rules), 'expected{} variable does'\
                ' not contain expected values for all rules. '.format(ii)
            for kk, rule_text in enumerate(self.rules_text):
                rule = self.rules[rule_text]
                self.assertEqual(rule.evaluate(inputs), expected[kk])


class RulesTest(unittest.TestCase):
    def setUp(self):
        # Define pore-like and a strand-like metrics
        self.pore = {'read_class': 'pore', 'range_current': 5,
                     'median_current': 190, 'median_sd': 1.2,
                     'is_saturated': False}

        self.strand = {'read_class': "'user1'", 'pore_before': 180,
                   'range_current': 30, 'is_saturated': False,
                   'median_current': 70, 'median_sd': 2,
                   'strand_duration': 10,
                   'num_events': 120}

        # Create rules to classify strands and pores
        rule1_text = "(median_current,lt,150) & " \
                     "(median_current$left,gt,median_current) & " \
                     "(median_current$right,gt,median_current) & " \
                     "(range_current, gt, 10)"
        self.rule1 = Rule(rule1_text, 'strand')

        rule2_text = "(median_current,lt,500) & " \
                     "(median_current, gt, 150)"
        self.rule2 = Rule(rule2_text, 'pore')

    def test_000_requirements(self):
        """ Check that the rule understands the keywords it needs to evaluate

        We are going to check:
            - A rule with several requirements and subrules joined by '&'
            - A rule with a single requirement but several subrules
        """
        required1 = {'median_current', 'range_current'}
        self.assertItemsEqual(self.rule1.requires, required1)
        required2 = {'median_current'}
        self.assertItemsEqual(self.rule2.requires, required2)

    def test_001_evaluations(self):
        """ Check that the rules evaluate correctly the reads.

        We will check:
            - A single pore metric is detected by the pore rule
            - A metric with pore at left and right and strand is detected by the strand rule
        """
        input1 = [self.pore]
        self.assertFalse(self.rule1.evaluate(input1))
        self.assertTrue(self.rule2.evaluate(input1))

        input2 = [self.pore, self.strand, self.pore]
        self.assertTrue(self.rule1.evaluate(input2))
        self.assertFalse(self.rule2.evaluate(input2))


class ReadRulesYamlTest(unittest.TestCase):

    def setUp(self):
        self.simple_config = [ 'pore   = (median_current,lt ,300)&(foo,eq,"bar")',
                               'notpore  = (median_current, gt , 350) &(5,eq,5)' ]
        self.left_right_config = [
            'strand =  (median_current,lt,400) & (median_current$left,gt,median_current) & (median_current$right,gt,median_current)',
            'pore     = (median_current,lt,500)',
            'block    = (median_current$left,gt,median_current) & (median_current$right,gt,median_current)',
        ]

        self.pore_before_after_config = [ r.replace(
            'median_current$left', 'pore_before').replace(
            'median_current$right', 'pore_after') for r in self.left_right_config ]

        self.metrifier = MockMetrics('median_current', 'foo', 'is_saturated', 'is_multiple', 'is_off')

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print '* ReadRules'

    def test_000_properties(self):
        classifier = ReadRules(self.metrifier, self.simple_config, time_since=None)
        self.assertEqual(classifier.n_reads, 3, 'Classifier does not declare three reads.')

        actual = classifier.requires
        expected = ['median_current', 'foo', 'is_saturated', 'is_multiple', 'is_off']
        self.assertItemsEqual(actual, expected, 'Classifier does not declare required metrics.')


    def test_001_basic_classification(self):
        classifier = ReadRules(self.metrifier, self.simple_config, time_since=None)

        reads = (
            ('unclassed', {'median_current':150, 'foo':10}),
            ('pore', {'median_current':150, 'foo':'bar'}),
            ('unclassed', {'median_current':150, 'foo':10}),
            ('notpore', {'median_current':400, 'foo':10}),
            ('unclassed', {'median_current':150, 'foo':10}),
        )
        expected, metrics = zip(*reads)
        actual = [x['class'] for x in classifier.process_reads(metrics)]
        self.assertListEqual(actual, list(expected), 'Classifier does not produce correct classes')

    def test_002_left_right_classification(self):

        reads = (
            ('pore', {'median_current':100}),
            ('unclassed', {'median_current':600}),
            ('block', {'median_current':550}),
            ('unclassed', {'median_current':600}),
            ('pore', {'median_current': 450}),
            ('strand', {'median_current': 300}),
            ('pore', {'median_current': 450}),
            ('pore', {'median_current':100}),
        )
        expected, metrics = zip(*reads)

        classifier = ReadRules(self.metrifier, self.left_right_config, time_since=None)
        actual = [x['class'] for x in classifier.process_reads(metrics)]
        print actual
        print expected
        self.assertListEqual(actual, list(expected), 'Classifier does not produce correct classes with $left/right sigils')

        classifier = ReadRules(self.metrifier, self.pore_before_after_config, time_since=None)
        actual = [x['class'] for x in classifier.process_reads(metrics)]
        self.assertListEqual(actual, list(expected), 'Classifier does not produce correct classes with pore_before/after')

if __name__ == "__main__":
    unittest.main()

