import unittest
import tempfile
from pathlib import Path
import numpy as np
from analysis.plot_saved_results import plot_metric

class TestPlot(unittest.TestCase):
    def test_plot(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"sample.npz";out=Path(d)/"plot.png"
            np.savez(path,results_exp_ret=np.ones((5,3,30)))
            plot_metric(path,"results_exp_ret",out,stride=5)
            self.assertTrue(out.is_file())
    def test_invalid_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"sample.npz"
            np.savez(path,results_exp_ret=np.ones((4,3,10)))
            with self.assertRaises(ValueError):plot_metric(path,"results_exp_ret",Path(d)/"x.png")
    def test_stride(self):
        with self.assertRaises(ValueError):plot_metric("missing.npz","results_exp_ret","out.png",0)
