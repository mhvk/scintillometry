# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of phaseutils sources."""
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.time import Time

from ..generators import NoiseGenerator
from ..pfb import PolyphaseFilterBank


class TestBasics:
    def setup(self):
        self.nh = NoiseGenerator(shape=(32 * 2048,),
                                 start_time=Time('2010-01-01'),
                                 sample_rate=1.*u.kHz, seed=12345,
                                 samples_per_frame=8, dtype='f8')
        self.nc = NoiseGenerator(shape=(32 * 2048,),
                                 start_time=Time('2010-01-01'),
                                 sample_rate=1.*u.kHz, seed=12345,
                                 samples_per_frame=8, dtype='c16')
        self.n_tap = 4
        self.n_chan = 2048
        n = self.n_tap * self.n_chan
        r = 2.
        x = r * (np.arange(n) / n * 2. - 1.)
        self.pfb = (np.sinc(x) * np.hamming(4 * 2048)).reshape(4, 2048)

    def test_understanding(self):
        """Stepping or frequency selection should give same answer.

        Often think of PFB as multiplying with an array that is, e.g.,
        4 times larger, then FFTing and taking every 4th frequency.

        But this is equivalent to, after multiplication, summing 4
        subsequent arrays and taking the FT of that.
        """
        self.nh.seek(0)
        # First check for real data.
        d = self.nh.read(5 * 2048).reshape(-1, 2048)
        hd = self.pfb * d[:4]
        ft1_hd = np.fft.rfft(hd.ravel())[::4]
        ft2_hd = np.fft.rfft(hd.sum(0))
        assert_allclose(ft1_hd, ft2_hd)

        # Check actual implementation.
        pfb = PolyphaseFilterBank(self.nh, self.pfb)
        ft_pfb = pfb.read(2)
        assert_allclose(ft_pfb[0], ft2_hd)
        assert_allclose(ft_pfb[1], np.fft.rfft((self.pfb * d[1:]).sum(0)))

    def test_understanding_complex(self):
        # Check above holds for complex too.
        self.nc.seek(0)
        c = self.nc.read(4 * 2048).reshape(-1, 2048)
        hc = self.pfb * c
        ft1_hc = np.fft.fft(hc.ravel())[::4]
        ft2_hc = np.fft.fft(hc.sum(0))
        assert_allclose(ft1_hc, ft2_hc)

        # And check actual implementation.
        pfb = PolyphaseFilterBank(self.nc, self.pfb)
        ft_pfb = pfb.read(1)[0]
        assert_allclose(ft_pfb, ft2_hc)
