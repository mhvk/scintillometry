import numpy as np


from .base import PaddedTaskBase
from .channelize import Channelize


class PolyphaseFilterBank(Channelize):
    """Channelize a time stream using a polyphase filter bank.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    response : `~numpy.ndarray`
        Polyphase filter.  The first dimension is taken to be the
        number of taps, and the second the number of channels.
    samples_per_frame : int, optional
        Number of complete output samples per frame (see Notes).  Default: 1.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).
    """
    def __init__(self, ih, response, samples_per_frame=None,
                 frequency=None, sideband=None):
        n_tap, n = response.shape
        pad = (n_tap - 1) * n
        assert pad % 1 == 0
        if samples_per_frame is not None:
            samples_per_frame = samples_per_frame * n
        self.padded = PaddedTaskBase(ih, pad_start=pad//2, pad_end=pad//2,
                                     samples_per_frame=samples_per_frame)
        self.padded.task = self.ppf
        self._response = response
        self._n_tap = n_tap
        super().__init__(self.padded, n, self.padded.samples_per_frame // n,
                         frequency=frequency, sideband=sideband)

    def ppf(self, data):
        response = self._response
        n_tap, n = response.shape
        data = data.reshape((-1, n) + data.shape[1:])
        result = np.empty((data.shape[0] + 1 - n_tap,) + data.shape[1:],
                          data.dtype)
        # TODO: use stride tricks to do this in one go.
        for i in range(data.shape[0] + 1 - n_tap):
            result[i] = (data[i:i+n_tap] * self._response).sum(0)
        return result.reshape((-1,) + result.shape[2:])
