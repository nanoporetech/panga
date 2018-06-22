Read Builder
--------------

A somewhat generic system for classifying the channel state throughout the
course of an experiment.

Here, a *read* is a contiguous run of events/raw data points that correspond to an
identifiable channel state.

We follow the same three-step process used in previous implementations, but allow
for some useful additions.

 1. **Determine read boundaries:** Calculates start and end coordinates of
    reads, and outputs reads as raw or event data. Naively, neighbouring reads
    are ultimately expected to be assigned different classes.
 
 2. **Calculate metrics on reads:** Takes event (or raw) data provided by
    step 1. and calculates summary statistics (of arbitrary complexity). May
    optionally take exogenous data concerning the state of a channel, e.g.
    saturation status.
    
 3. **Classify reads:** Takes the matrix of metrics for a read (and possibly the
    metrics of surrounding reads) to output an additional *metric*: the
    classification.

The smart cookie notices that this system is rather *backwards*: one would hope
to label read boundaries by running a classifier over a channel and outputting
boundaries where there is a state change. The above does not exclude this;
step 1. could be arbitrarily complex. The framework requires only that it
output read data (and optionally key-value pairs of meta data).

The key differences between the current framework and previous incarnations are:

  * Clear separation of the three steps.
  * The ability to calculate classes based on metrics of surrounding reads.
  * Defined interface to each step such that the methodology used in each can
    be changed depending on the application.
  * Low memory use: reads are processed individually rather than storing all
    data for a channel in memory.
  * Multiprocessing across channels.

Installation
------------

Panga installs into a virtual environment. Installation should should be as simple as running  
    Make install 

On Ubuntu, you will need to install the virtualenv and python-dev packages.
    sudo apt-get install python-virtualenv python-dev 

Running read_builder
--------------------

Panga read_builder is installed into a virtual environment. To run it first activate your environment
    source panga_directory/venv/bin/activate 


To analyse Minknow reads use:
    read_builder --fast5 path_to_bulk_hdf --config $(panga_config_dir)/standard_minknow_classes.yml --outpath read_builder --summary_file read_summary.txt 

Example configs for more advanced analyses such as redetection of read boundaries, read metrics and classifications can be found in the panga config directory:
    ls $(panga_config_dir)
