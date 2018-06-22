import unittest
import numpy as np
from panga.algorithm.deltasplit import read_bounds_from_delta


class DeltaSplitterTest(unittest.TestCase):
    """Test if delta splitter works
    """
    @classmethod
    def setUpClass(cls):
        print '* DeltaSplitter'

    def setUp(self):
        self.data = np.array([300, 301, 299,
                              200, 199, 190, 220,
                              300, 325, 350, 375,400,
                              441, 400,
                              200,
                              100, 101], dtype=float)
        self.expected = [(0, 3), (3, 7), (7, 12), (12, 14), (14, 15), (15, 17)]

    def test_000_python_delta_splitter(self):
        """ Check that the subrules recognize which keys it needs to provide
        """
        bounds = read_bounds_from_delta(self.data, use_cython=False)
        assert len(self.expected) == len(bounds)
        assert np.all(np.array(self.expected) == np.array(bounds))

    def test_001_cython_delta_splitter(self):
        """ Check that the subrules recognize which keys it needs to provide
        """
        bounds = read_bounds_from_delta(self.data, use_cython=True)
        assert len(self.expected) == len(bounds)
        assert np.all(np.array(self.expected) == np.array(bounds))

if __name__ == "__main__":
    unittest.main()
