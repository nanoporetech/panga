import itertools
from panga.stage.base import StageBase

__all__ = ('PassStage',)


class PassStage(StageBase):

    def __init__(self, splitter, classifier):
        """Simple pass through stage."""
        super(PassStage, self).__init__(splitter, classifier)

    def _process_reads(self, reads, metrics):
        return reads, metrics
