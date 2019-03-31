# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of phaseutils sources."""
import numpy as np
from numpy.testing import assert_allclose

from .. import pfb


class TestBasics:
    def test_understanding(self):
        """Stepping or frequency selection should give same answer.

        Normally think of PFB as multiplying with an array that is, e.g.,
        4 times larger, then FFTing and taking every 4th frequency.

        But this is equivalent to, after multiplication, summing 4
        subsequent arrays and taking the FT of that.
        """
        d = np.random.normal(0., 1., 8192)
        h = pfb.sinc_hamming(4, 2048)
        hd = h * d
        f1 = np.fft.rfft(hd)[::4]
        f2 = np.fft.rfft(hd.reshape(4, -1).sum(0))
        assert_allclose(f1, f2)
        # Check this holds for complex too.
        c = np.random.normal(0., 1., 16384).view(np.complex128)
        hc = h * c
        f1 = np.fft.fft(hc)[::4]
        f2 = np.fft.fft(hc.reshape(4, -1).sum(0))
        assert_allclose(f1, f2)
