import Queue
import argparse
import funcsigs
import functools
import inspect
import itertools
import json
import logging
import numpy as np
import matplotlib; matplotlib.use('Agg', warn=False)  # enforce non-interactive backend
import multiprocessing
import os
import sys
import traceback
import yaml

from cStringIO import StringIO
from copy import deepcopy
from fast5_research.fast5_bulk import BulkFast5
from panga import cmdargs

from panga import split, metric, classify, stage, accumulate, conclude
from panga.util import add_prefix

logger = logging.getLogger(__name__)

default_config = """
read_builder:
    components:
        splitter:
            RandomData:
        metrifier:
            StandardMetrics:
        classifier:
            SimpleClassifier:
        second_stage:
            PassStage:
        accumulator:
            MetricSummary:
"""


# Build a dictionary of available component types
single_component_modules = dict(zip(  # use only one of these
        ('splitter', 'metrifier', 'classifier', 'second_stage'),
        (split, metric, classify, stage))
)
multi_component_modules = dict(zip(  # use one of more of these
        ('accumulator', 'concluder'),
        (accumulate, conclude))
)
component_modules = dict(single_component_modules.items() + multi_component_modules.items())
required_components = set(component_modules.keys()) - set(['concluder'])  # concluders are optional
components = {
    name:[x[0] for x in inspect.getmembers(component, inspect.isclass) if 'Base' not in x[0] ]
    for name, component in component_modules.items()
}


def make_resolver(my_component):
    """Create and argparse.action to set a component class."""
    class MyResolver(argparse.Action):
        component = my_component
        def __call__(self, parser, namespace, values, option_string=None):
            if isinstance(values, list):
                setattr(namespace, self.dest, [getattr(self.component, v) for v in values])
            else:
                setattr(namespace, self.dest, getattr(self.component, values))
    return MyResolver


class SetReadRulesAction(argparse.Action):
    """Set classifier component when read rule config is given."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, 'classifier', classify.ReadRules)
        if not os.path.exists(values):
            raise RuntimeError("File/path for '{}' does not exist, {}".format(self.dest, values))


class ParseJsonDictAction(argparse.Action):
    """Parse JSON str into a dict."""
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            meta_dict = json.loads(values)
        except:
            raise RuntimeError("{} is not a valid JSON string".format(values))
        if not isinstance(meta_dict, dict):
            raise RuntimeError("{} did not parse as a dict.".format(values))

        setattr(namespace, self.dest, meta_dict)

class MoreHelpAction(argparse.Action):
    """Parse JSON str of a list of lists into a tuple of tuples."""
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            print '\nList of panga components which can be used in read_builder:\n'
            for component, klasses  in components.items():
                print '{}: \n\t{}\n'.format(component, '\n\t'.join(klasses))
        else:  # fetch docstring of a component
            klass = None
            for component, klasses in components.items():
                if values in klasses:
                    klass = getattr(component_modules[component], values)
            if klass is None:
                raise KeyError("{} is an unrecognised class.".format(values))
            # print initialiser doc string which should show params and their meaning
            print klass.__init__.__doc__
        parser.exit()

def get_argparser():
    parser = argparse.ArgumentParser(
        description='Analyse channel states to output read classification summary.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--more_help', nargs='?', default=None, action=MoreHelpAction,
                        help='Show list of components. If a specific component is provided, print its doc string.')
    parser.add_argument('--channels', default=None, metavar='Channel List', action=cmdargs.ExpandRanges, help='Channels to process, default is all channels.')
    parser.add_argument('--max_time', default=np.inf, help='Stop processing reads after a certain time.')
    parser.add_argument('--jobs', default=1, type=int, help='Number of processes to use.')
    parser.add_argument('--fast5', action=cmdargs.FileExists, help='Bulk fast5 file for input.')
    parser.add_argument('--outpath', help='Output directory (should not already exist)')
    parser.add_argument('--summary_file', help='Metric output summary filename.')
    parser.add_argument('--prefix', type=str, default='', help='prefixed to output files.')
    parser.add_argument('--meta', type=str, default=None, action=ParseJsonDictAction,
                        help='JSON meta string (e.g. \'{"run":"asd123"}\'.')
    parser.add_argument('--input_summary', type=str, action=cmdargs.FileExists, help='Input summary for splitting/metrics')
    parser.add_argument('--config', action=cmdargs.FileExists, help='Read_builder config.')
    parser.add_argument('--config_out', type=str, default=None, help='Write config.')
    parser.add_argument('--config_only', action='store_true',
                        help='Use only config options, use this for rerunning an outputed config.')

    return parser

def load_yaml_config(yaml_conf):
    try:
        if hasattr(yaml_conf, 'readline'):
            conf = yaml.load(yaml_conf)
        else:
            with open(yaml_conf, 'r') as fh:
                conf = yaml.load(fh)
    except:
        raise RuntimeError('Could not parse your yaml file {}, check it is correctly formatted'.format(yaml_conf))
    return conf

def resolve_rules_config(config_dict):
    """If a ReadRules rules arg points to a file, load the rules from that file"""
    if 'ReadRules' in config_dict['components']['classifier']:
        read_rules_args = config_dict['components']['classifier']['ReadRules']
        if not 'rules' in read_rules_args:
            raise KeyError('ReadRules requires a rules option')
        rules = read_rules_args['rules']
        if isinstance(rules, str):  # assume this points to a yaml file
            logger.info('Loading ReadRules config from : {}.'.format(rules))
            loaded = load_yaml_config(rules)
            if 'parameters' in loaded.keys() and 'rules_in_execution_order' in loaded['parameters']:
                rules_lines = loaded['parameters']['rules_in_execution_order']
                read_rules_args['rules'] = rules_lines
            else:
                raise KeyError('ReadRules config could not be read from {}'.format(rules))


def get_class_arg_store(arg_store):

    components_sec = 'components'
    if components_sec not in arg_store:
        raise KeyError('A components section is required in your config')

    # copy arg_store into new dict where component keys are classes instead of
    # strings of class names
    class_arg_store = deepcopy(arg_store)

    if not set(arg_store[components_sec]).issuperset(required_components):
        raise KeyError('Not all required components were specified. Missing {}'.format(
                        required_components - set(arg_store[components_sec])))

    for component in arg_store[components_sec]:
        if component not in components:
            raise KeyError('Unknown component type {}, use one of {}'.format(component, components.keys()))
        # check we only have 1 splitter, metrifier and classifier
        if component in single_component_modules:
            keys = arg_store[components_sec][component].keys()
            if len(keys) > 1:
                raise RuntimeError('Only 1 {} should be specified, found {}'.format(component, keys))

        for klass_name in arg_store[components_sec][component]:
            if klass_name not in components[component]:
                raise KeyError('Unknown {} class `{}`, use one of {}'.format(component, klass_name, components[component]))
            klass = getattr(component_modules[component], klass_name)
            # index class options by class instead of class name
            class_arg_store[components_sec][component][klass] = arg_store[components_sec][component][klass_name]
            del class_arg_store[components_sec][component][klass_name]

    return class_arg_store

def process_args(arguments=None):
    """ Read and do an initial manipulation of inputs into a dict.

    :param arguments: list of arguments
    :return: dict, argstore of options organised by component class.
    :return: str,  all options as yaml str.
    """
    # load cmd line args and copy to a dict
    cmdline_args = get_argparser().parse_args(arguments).__dict__

    configs = {'config':None, 'config_out': None}
    for key in configs:
        if key in cmdline_args:
            configs[key] = cmdline_args[key]
            del cmdline_args[key]

    # if a config was specified on the command line, load it
    section_name = 'read_builder'
    if configs['config'] is not None:
        arg_store = load_yaml_config(configs['config'])[section_name]
    else:
        arg_store = load_yaml_config(StringIO(default_config))[section_name]

    # update arg_store with cmdline args (without config names)
    # except if we are running from an outputed config, where everything we need
    # should be in the config
    if not cmdline_args['config_only']:
        arg_store.update(cmdline_args)

    # check if we need to load a ReadRules config (a Minknow read classifier yaml)
    resolve_rules_config(arg_store)

    # check components and replace component class names with actual classes
    class_arg_store = get_class_arg_store(arg_store)

    # add 'config_out' to class_arg_store so we can later write a yml
    class_arg_store['config_out'] = configs['config_out']

    # create yaml string which can be later written out
    yaml_conf_out = yaml.dump({section_name:arg_store})

    return class_arg_store, yaml_conf_out


def get_class_args(cls, class_args, arg_store, presets=dict()):
    """Resolve the arguments to be used for a component.

    :param cls: the component class
    :param cls_args: dict of arguments specific to this class, will be used in
           preference to args in `arg_store`. If None, will be ignored.
    :param arg_store: namespace containing all possible arguments
    :param preset: preset argument values, may correspond to postional or
        keyword arguments. Will be used in preference to any others..
    """
    params = funcsigs.signature(cls.__init__).parameters
    del params['self']  # strip out self from the args
    args = []
    kwargs = dict()
    for name, param in params.items():
        value = None
        if name in presets.keys():  # presets first
            value = presets[name]
        elif class_args is not None and name in class_args:  # then class args
            value = class_args[name]
        elif name in arg_store:  # finally arg store
            value = arg_store[name]
        if param.default is param.empty:  # positional arg
            if value is None:
                raise ValueError(
                    'Component {} requires --{} argument.'.format(cls.__name__, name))
            args.append(value)
        else:  # keyword argument
            if value is None:
                value = param.default
            kwargs.update({name: value})
    return args, kwargs


def get_splitter(arg_store, channel):
    presets = {'channel':channel}
    splitter, splitter_args = arg_store['components']['splitter'].items()[0]
    args, kwargs = get_class_args(splitter, splitter_args, arg_store, presets)
    return splitter(*args, **kwargs)


def get_metrifier(arg_store, splitter):
    presets = {'splitter':splitter}
    metrifier, metrifier_args = arg_store['components']['metrifier'].items()[0]
    args, kwargs = get_class_args(metrifier, metrifier_args, arg_store, presets)
    return metrifier(*args, **kwargs)


def get_classifier(arg_store, metrifier):
    presets = {'metrifier':metrifier}
    classifier, classifier_args = arg_store['components']['classifier'].items()[0]
    args, kwargs = get_class_args(classifier, classifier_args, arg_store, presets)
    return classifier(*args, **kwargs)


def get_second_stage(arg_store, splitter, classifier):
    presets = {'splitter':splitter, 'classifier':classifier}
    second_stage, second_stage_args = arg_store['components']['second_stage'].items()[0]
    args, kwargs = get_class_args(second_stage, second_stage_args, arg_store, presets)
    return second_stage(*args, **kwargs)


def get_accumulators(arg_store):
    presets = {}
    accumulators = []
    for accumulator, accumulator_args in arg_store['components']['accumulator'].items():
        args, kwargs = get_class_args(accumulator, accumulator_args, arg_store, presets)
        accumulators.append(accumulator(*args, **kwargs))
    return accumulators


def get_concluders(arg_store, channel):
    presets = {'channel':channel}
    concluders = []
    if 'concluder' in arg_store['components']:
        for concluder, concluder_args in arg_store['components']['concluder'].items():
            args, kwargs = get_class_args(concluder, concluder_args, arg_store, presets)
            concluders.append(concluder(*args, **kwargs))
    return concluders


def get_pipeline(args, channel):
    splitter = get_splitter(args, channel)
    metrifier = get_metrifier(args, splitter)
    classifier = get_classifier(args, metrifier)
    second_stage = get_second_stage(args, splitter, classifier)
    return splitter, metrifier, classifier, second_stage


def process_channel_queue(args, channel, queue):
    """Process a channel putting results into a shared queue.

    :param args: namespace of miscellaneous stuff, see entrypoint args.
    :param channel: numeric channel identifier.
    :param queue: shared queue.
    """
    for read_metrics in process_channel(args, channel):
        #TODO: buffered puts? might decrease overheard
        queue.put(read_metrics)

def process_channel(args, channel):
    """Process a channel yielding metrics for reads

    :param args: namespace of miscellaneous stuff, see entrypoint args.
    :param channel: numeric channel identifier.
    """
    logger.info('Processing channel {}.'.format(channel))
    # Set up pipeline
    read_generator, metric_calculator, classifier, second_stage = get_pipeline(args, channel)
    concluders = get_concluders(args, channel)

    # Get read data - tee because we want the reads to run second stage
    reads_0, reads_1 = itertools.tee(read_generator.reads())

    # Calculate metrics
    metrics = metric_calculator.process_reads(reads_0)

    # Classify - augments read metrics with 'class'
    class_metrics = classifier.process_reads(metrics)

    # Run second stage analysis to manipulate reads and metrics
    final_reads, final_metrics = second_stage.process_reads(reads_1, class_metrics)

    # Output
    for i, (read, read_metrics) in enumerate(itertools.izip(final_reads, final_metrics)):
        # run concluders before yielding metrics so anything they inject is
        # printed in the summary
        for concluder in concluders:
            concluder.process_read(read, read_metrics)
        yield read_metrics

    for concluder in concluders:
        logger.info('Running {} for channel {}.'.format(concluder, channel))
        concluder.finalize()


def except_functor(function, *args, **kwargs):
    """Wrapper for worker functions, running under multiprocessing, to better
    display tracebacks when the functions raise exceptions."""
    try:
        return function(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def accumulate_channels(args):
    #TODO: drop this idiom into untangled
    if args['jobs'] == 1:
        logging.info('Running with single-process iterator.')
        for channel in args['channels']:
            for item in process_channel(args, channel):
                yield item
    else:
        logging.info('Running with {}-process iterator.'.format(args['jobs']))
        pool = multiprocessing.Pool(args['jobs'])
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        promises = [
            pool.apply_async(functools.partial(
                except_functor, process_channel_queue, args, channel, queue)
            ) for channel in args['channels']
        ]
        pool.close()
        while True:
            # check for any exceptions
            try:
                [p.get(0.01) for p in promises]
            except Exception as e:
                if not isinstance(e, multiprocessing.TimeoutError):
                    raise e
            # try to get any item, but dont block. If nothing
            # in queue, exit if all promises resolved.
            try:
                item = queue.get(False, 0.01)
            except Queue.Empty:
                if all(x.ready() for x in promises):
                    break
            else:
                yield item
        pool.join()


def main(args=None):
    logging.basicConfig(format='[%(asctime)s - %(name)s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

    if args is None:
        args = sys.argv[1:]

    # process args and get yaml string of all options
    args, yaml_conf_out = process_args(args)

    logging.debug('args are: {}'.format(args))

    logger.info("Will stop after {} seconds of expt time.".format(args['max_time']))

    # Multiple components will write here
    if 'outpath' in args and args['outpath'] is not None:
        os.mkdir(args['outpath'])

    # save the config
    if args['config_out'] is not None:
        path = add_prefix(args['config_out'], args['prefix'])
        if 'outpath' in args and args['outpath'] is not None:
            path = os.path.join(args['outpath'], path)
        with open(path, 'w') as fh:
            fh.write(yaml_conf_out)

    # Get channel range from bulk if it was not specified
    if args['channels'] is None:
        with BulkFast5(args['fast5'], 'r') as f5:
            args['channels'] = list(f5.channels)
    else:
        args['channels'] = list(args['channels'])

    # Test pipeline can be constructed
    read_generator, metric_calculator, classifier, second_stage = get_pipeline(args, args['channels'][0])
    logger.info('Splitter      : {}.'.format(read_generator))
    logger.info('Metrifier     : {}.'.format(metric_calculator))
    logger.info('Classifier    : {}.'.format(classifier))
    logger.info('SecondStage   : {}.'.format(second_stage))

    # Accumulators gather results from individual channels
    accumulators = get_accumulators(args)
    logger.info('Accumulators  : {}.'.format(accumulators))
    for read_metrics in accumulate_channels(args):
        for accumulator in accumulators:
            accumulator.process_read(read_metrics)

    # Finish up accumulators
    for accumulator in accumulators:
        accumulator.finalize()


if __name__ == '__main__':
    main()
