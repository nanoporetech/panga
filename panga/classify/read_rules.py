from ConfigParser import ConfigParser
import re
import operator
import string
import yaml
from collections import namedtuple

from panga.classify.simple import StandardClassifier

Key = namedtuple('Key', 'keyword position')


class SubRule(object):
    """A Subrule will receive a string with structure:

       (part1, operand, part2)

    where part1 and part2 will be compared using the operand. Either part
    could be a number, a pure string (such as 'pore', 'user1') or a keyword
    refering to an attribute of the metrics (e.g. median_current,
    pore_before).

    The result will be an object that has done the parsing of the parts and
    the operand, and can be used to apply the rule it represents.
    """
    _op_dict = {
               'gt': operator.gt,
               'lt': operator.lt,
               'eq': operator.eq,
               'ne': operator.ne,
               '+': operator.add,
               '-': operator.sub,
               '*': operator.mul,
               '/': operator.div
    }

    def __init__(self, string):
        """Initialize the SubRule object using a string. Strip tabs and spaces,
        parse the left hand side, right hand side and operator.
        :param string: string of the form (part1, operator, part2), often used
                       in our config files
        """
        string = re.sub('\s', '', string)
        match = re.match('\((.*),(.*),(.*)\)', string)
        if match:
            lhs, oper, rhs = match.groups()
        else:
            lhs, oper, rhs = None, None, None
            raise SyntaxError('Rule: "{}" could not be parsed. '.format(string))

        self.lhs = self._parse_attr(lhs)
        self.rhs = self._parse_attr(rhs)
        self.oper = self._op_dict[oper]
        self.string = string
        self.__requires = None
        self.__requires_side = None

    def __repr__(self):
        return 'SubRule({})'.format(self.string)

    def _parse_attr(self, string):
        """Parse a string in a way that can be easily interpreted as either
        a number or a string.

        :param string: string to be parsed
        :return: float if it can be converted, string otherwise
        """
        key_translator = { 'pore_before' : 'median_current$left',
                           'pore_after' : 'median_current$right',
                           'strand_duration': 'duration',
                           'median_before': 'median_current$left',
                           'median': 'median_current',
                           'range': 'range_current',
                           'event_count': 'num_events',
        }

        if string in key_translator:
            string = key_translator[string]

        # If it is a number, return a number
        try:
            return float(string)
        except ValueError:
            pass

        # If it is literally a string to compare with, e.g. "'pore'", return the
        # string without the ''
        if string.startswith("'") and string.endswith("'"):
            return string.replace("'", "")
        if string.startswith('"') and string.endswith('"'):
            return string.replace('"', '')

        # If it's written in the new way 'median_current$0' or 'range_current$1'
        try:
            return Key(*string.split('$'))
        except TypeError:
            pass

        # If it is a subrule within a subrule, for example:
        #     median_current + 2 * range
        # return a Subrule
        # We need to be carefull to keep the priorities of operations, in
        # this case keeping the '*' and '/' till the end, so they are evaluated
        # first after recurrence
        for op in ['+', '-', '*', '/']:
            if op in string:
                try:
                    split = string.split(op)
                    split = [ key_translator[s] if s in key_translator else s for s in split ]
                    return SubRule('({}, {}, {})'.format(split[0], op, split[1]))
                except SyntaxError:
                    pass


        # The last possible case is the keyword did not have a '$', which means
        # it refers to position 1 even if not explicitly stated, i.e. a good old
        # 'median_current' .
        return Key(string, 'centre')

    @property
    def requires(self):
        """Required attributes are all those participating in the comparisons
        that are not numbers or literally strings to compare with. They
        should be part of the metrics, and will be called when evaluating.

        :return: a list of the required keywords.
        """
        if self.__requires is not None:
            return self.__requires

        required = set()
        self.__requires_side = {side:False for side in ('left', 'right')}
        for element in self.lhs, self.rhs:
            if isinstance(element, Key):
                required.add(element.keyword)
                for side in ('left', 'right'):
                    self.__requires_side[side] |= element.position == side
            # Recursivity: there is a subrule in this side of this rule
            if isinstance(element, SubRule):
                required = required.union(element.requires)
                for side in ('left', 'right'):
                    self.__requires_side[side] |= element._requires_side(side)
        self.__requires = required
        return required

    def _requires_side(self, side):
        """Do any metrics refer to reads in the left/right position."""
        if self.__requires_side is not None:
            return self.__requires_side[side]
        else:
            _ = self.requires
            return self.__requires_side[side]

    @property
    def requires_right(self):
        """Do any metrics refer to reads in the right position."""
        return self._requires_side('right')

    @property
    def requires_left(self):
        """Do any metrics refer to reads in the left position."""
        return self._requires_side('left')

    def evaluate(self, metrics):
        """ Evaluate the comparison of the rule. If the elements are keywords
        meant for the metric, use them as such.

        :param metrics: mapping object containing the metrics of the reads.
        :return: boolean indicating if the reads given pass the current subrule
        """

        # Note the next line copies references to unmutable objects
        centre = len(metrics) // 2
        left = centre - 1
        right = centre + 1
        pos_dict = {'centre': centre, 'left': left, 'right': right}

        # Probably this should be refractored.
        lhs, rhs = self.lhs, self.rhs
        if isinstance(lhs, Key):
            lhs = metrics[pos_dict[lhs.position]][lhs.keyword]
        elif isinstance(lhs, SubRule):
            lhs = lhs.evaluate(metrics)

        if isinstance(rhs, Key):
            rhs = metrics[pos_dict[rhs.position]][rhs.keyword]
        elif isinstance(rhs, SubRule):
            rhs = rhs.evaluate(metrics)
        return self.oper(lhs, rhs)


class Rule(object):
    def __init__(self, rule_string, rule_name):
        """
        :param rule_string:
        :param rule_name: Anything that identifies the rule, it can be a name, an ID
                   number, a hash, ...
        """
        stripped = re.sub('\s', '', rule_string)
        self.subrules = [SubRule(ii) for ii in stripped.split('&')]
        self.string = rule_string
        self.rule_name = rule_name

    def __repr__(self):
        return "Rule({})".format(self.string)

    @property
    def requires(self):
        return reduce(set.union, (srule.requires for srule in self.subrules))

    def _requires_side(self, side):
        return any(srule._requires_side(side) for srule in self.subrules)

    @property
    def requires_right(self):
        """Do any metrics refer to reads in the right position."""
        return self._requires_side('right')

    @property
    def requires_left(self):
        """Do any metrics refer to reads in the left position."""
        return self._requires_side('left')

    def evaluate(self, metrics):
        return all((rule.evaluate(metrics) for rule in self.subrules))


class ReadRules(StandardClassifier):

    def __init__(self, metrifier, rules,
                 to_inject=(('median_current', -1, '_before', 0),
                            ('median_current', 1, '_after', 0)),
                 time_since=('strand',), recovered_classes=None, state_classes=None,
                 ):
        """Classify reads according to simple rules in a file.

        :param rules: list of str, rules in execution order.
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
            to_inject=(('median_current', -1, '_before', 0)) would inject the new
            metric 'median_current_before' with the value of the
            'median_current' of the preceeding read.
        """
        self.rules = self.__parse_rules(rules)
        self.inject_time_since = time_since
        super(ReadRules, self).__init__(metrifier, class_metric=None, to_inject=to_inject,
                                        time_since=time_since, recovered_classes=recovered_classes,
                                        state_classes=state_classes)


    @property
    def requires(self):
        requires = super(ReadRules, self).requires
        rules_requires = list(reduce(set.union, (rule.requires for rule in self.rules)))
        requires = set(requires).union(rules_requires)
        return requires

    @property
    def n_reads(self):
        return 3


    def _classify_read(self, metrics):
        """Update the read_class by applying the rules to the metrics.

        :returns: The classification.
        """
        klass = 'unclassed'
        for rule in self.rules:
            # skip evaluation of rules which require metrics from surrounding reads
            if metrics[0] is None and rule.requires_left:
                continue
            if metrics[2] is None and rule.requires_right:
                continue
            if rule.evaluate(metrics):
                klass = rule.rule_name
                break
        return klass

    def __parse_rules(self, rules_lines):
        rules = []
        for line in rules_lines:
            class_name, rule_data = map(string.strip, line.split('='))
            rules.append(Rule(rule_data, class_name))
        return rules
