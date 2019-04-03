# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy.utils import lazyproperty


from .base import PaddedTaskBase
from .channelize import Channelize


__all__ = ['PolyphaseFilterBankSamples', 'PolyphaseFilterBank']


class PolyphaseFilterBankSamples(Channelize):
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
        if samples_per_frame is not None:
            samples_per_frame = samples_per_frame * n
        assert pad % 2 == 0
        self.padded = PaddedTaskBase(ih, pad_start=pad//2, pad_end=pad//2,
                                     samples_per_frame=samples_per_frame)
        self.padded.task = self.ppf
        self._response = response
        super().__init__(self.padded, n, self.padded.samples_per_frame // n,
                         frequency=frequency, sideband=sideband)
        self._reshape = ((self.padded._ih_samples_per_frame // n, n)
                         + self.ih.sample_shape)

    def ppf(self, data):
        data = data.reshape(self._reshape)
        result = np.empty((self.samples_per_frame,) + data.shape[1:],
                          data.dtype)
        # TODO: use stride tricks to do this in one go.
        n_tap = len(self._response)
        for i in range(data.shape[0] + 1 - n_tap):
            result[i] = (data[i:i+n_tap] * self._response).sum(0)
        return result.reshape((-1,) + result.shape[2:])


class PolyphaseFilterBank(PolyphaseFilterBankSamples):
    def __init__(self, ih, response, samples_per_frame=None,
                 frequency=None, sideband=None):
        super().__init__(ih, response=response,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband)
        self._ppf_fft = self._FFT(shape=self._reshape, dtype=self.ih.dtype)
        self._ppf_ifft = self._ppf_fft.inverse()

    @lazyproperty
    def _ft_response_conj(self):
        long_response = np.zeros(self._reshape[:2], self.ih.dtype)
        long_response[:self._response.shape[0]] = self._response
        long_response.shape = (long_response.shape
                               + (1,) * len(self.ih.sample_shape))
        fft = self._FFT(shape=long_response.shape, dtype=self.ih.dtype)
        return fft(long_response).conj()

    def ppf(self, data):
        data = data.reshape(self._reshape)
        ft = self._ppf_fft(data)
        ft *= self._ft_response_conj
        result = self._ppf_ifft(ft)
        # Remove padding, which has turned around.
        result = result[:result.shape[0]+1-self._response.shape[0]]
        # Reshape as timestream (channelizer will immediately undo this...).
        return result.reshape((-1,) + result.shape[2:])
