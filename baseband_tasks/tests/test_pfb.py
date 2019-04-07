# Licensed under the GPLv3 - see LICENSE
"""Tests of the polyphase filterbank module."""
import os

import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.time import Time

from ..generators import NoiseGenerator
from ..pfb import sinc_hamming, PolyphaseFilterBankSamples, PolyphaseFilterBank


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestSincHamming:
    def setup(self):
        a = np.loadtxt(os.path.join(
            test_data, 'bGDSP_U1_0032_T12_W095_get_pfb_coeffs.txt'))
        self.guppi_ppf = a.reshape(8, -1).T.reshape(12, 64)

    def test_guppi(self):
        model = sinc_hamming(12, 64, sinc_scale=0.95)
        assert_allclose(model, self.guppi_ppf)


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
        r = self.n_tap
        x = r * np.linspace(-0.5, 0.5, n, endpoint=False)
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

        # Check actual implementations.
        pfb = PolyphaseFilterBankSamples(self.nh, self.pfb)
        ft_pfb = pfb.read(2)
        assert_allclose(ft_pfb[0], ft2_hd)
        assert_allclose(ft_pfb[1], np.fft.rfft((self.pfb * d[1:]).sum(0)))
        pfb2 = PolyphaseFilterBank(self.nh, self.pfb)
        ft_pfb2 = pfb2.read(2)
        assert_allclose(ft_pfb2, ft_pfb)

    def test_understanding_complex(self):
        # Check above holds for complex too.
        self.nc.seek(0)
        c = self.nc.read(4 * 2048).reshape(-1, 2048)
        hc = self.pfb * c
        ft1_hc = np.fft.fft(hc.ravel())[::4]
        ft2_hc = np.fft.fft(hc.sum(0))
        assert_allclose(ft1_hc, ft2_hc)

        # And check actual implementation.
        pfb = PolyphaseFilterBankSamples(self.nc, self.pfb)
        ft_pfb = pfb.read(1)[0]
        assert_allclose(ft_pfb, ft2_hc)
        pfb = PolyphaseFilterBank(self.nc, self.pfb)
        ft_pfb = pfb.read(1)[0]
        assert_allclose(ft_pfb, ft2_hc)
