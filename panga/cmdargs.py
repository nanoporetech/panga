import argparse
import os
import re


class FileExists(argparse.Action):
    """Check if the input file exist."""

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            raise RuntimeError("File/path for '{}' does not exist, {}".format(self.dest, values))
        setattr(namespace, self.dest, values)


class ExpandRanges(argparse.Action):
    """Translate a str like 1,2,3-5,40 to [1,2,3,4,5,40]"""

    def __call__(self, parser, namespace, values, option_string=None):
        elts = []
        for item in values.replace(' ', '').split(','):
            mo = re.search(r'(\d+)-(\d+)', item)
            if mo is not None:
                rng = [int(x) for x in mo.groups()]
                elts.extend(list(range(rng[0], rng[1] + 1)))
            else:
                elts.append(int(item))
        setattr(namespace, self.dest, elts)
