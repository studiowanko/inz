# TODO: get how to import things in python better than this way: xd
import sys

sys.path.insert(0, '/mnt/c/inz/inz/')

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
try:
    from unittest import mock
except ImportError:
    import mock


class testMathFunctions(unittest.TestCase):

    def test_angle_delta(self):
        from wutsat.fun import mat_fun as mf
        import math as m
        beta = mf.find_angle_analytic(6371, 450, 20 * m.pi / 180)
        self.assertLess(beta, 1e+10)

    def test_correct_data_implementation(self):
        from wutsat.fun import mat_fun as mf
        import numpy as np
        latitudes, longitudes = [10, 20, 30, 40], [20, 30, 40, 50]
        lines = [[[latitudes], [longitudes]]]
        for i, polygon in enumerate(lines):
            poly_lats, poly_lons = polygon[0], polygon[1]
            for j, (la, lo) in enumerate(zip(poly_lats, poly_lons)):
                self.assertTrue(len(la) == len(lo) == np.shape(lines[i][0][j])[0] == np.shape(lines[i][0][j])[0])

    def test_incorrect_data_implementation(self):
        from wutsat.fun import mat_fun as mf
        import numpy as np
        latitudes, longitudes = [[[10], 20, 30, 40]], [20, [30], 40, 50]
        lines = [[[latitudes], [longitudes]]]
        for i, polygon in enumerate(lines):
            poly_lats, poly_lons = polygon[0], polygon[1]
            for j, (la, lo) in enumerate(zip(poly_lats, poly_lons)):
                self.assertFalse(len(la) == len(lo) == np.shape(lines[i][0][j])[0] == np.shape(lines[i][0][j])[0])


if __name__ == '__main__':
    unittest.main()
