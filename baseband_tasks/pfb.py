# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy.utils import lazyproperty


from .base import PaddedTaskBase
from .channelize import Channelize, Dechannelize


__all__ = ['PolyphaseFilterBankSamples', 'PolyphaseFilterBank']


def sinc_hamming(n_tap, n_sample, sinc_scale=1.):
    r"""Construct a sinc-hamming polyphase filter.

    The sinc-hamming filter is defined by

    .. math::  \frac{\sin(\pi x)}{\pi x}
               \left[0.54 - 0.46\cos\left(\frac{2\pi k}{N-1}\right)\right]
               \qquad x = n_{\rm tap} scale \left(\frac{n}{N} - 0.5\right),
               \qquad N=n_{\rm tap} n_{\rm sample}, \quad 0 \leq k \leq N-1

    Parameters
    ----------
    n_tap : int
        Number of taps of the polyphase filter.
    n_samples : int
        Number of samples to pass on to the FFT stage.
    sinc_scale : float
        Possible scaling for the sinc factor, to widen or narrow the channels.

    Examples
    --------
    Construct the CHIME and GUPPI baseband polyphase filter responses::

    >>> from baseband_tasks.pfb import sinc_hamming
    >>> chime_ppf = sinc_hamming(4, 2048)
    >>> guppi_ppf = sinc_hamming(12, 64, sinc_scale=0.95)
    """
    n = n_tap * n_sample
    x = n_tap * sinc_scale * np.linspace(-0.5, 0.5, n, endpoint=False)
    return (np.sinc(x) * np.hamming(n)).reshape(n_tap, n_sample)


class PolyphaseFilterBankSamples(Channelize):
    """Channelize a time stream using a polyphase filter bank.

    This version applies the polyphase filter in the time domain.

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
        # Note: cannot easily use Convolve since this doesn't convolve
        # every sample, but rather each n_chan samples.  In principle,
        # it could be done via a ReshapeSamples followed by a Convolve.
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
    """Channelize a time stream using a polyphase filter bank.

    This version applies the polyphase filter in the frequency domain.

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


class InversePolyphaseFilterBank(PaddedTaskBase):
    def __init__(self, ih, response, sn, n=None, samples_per_frame=None,
                 frequency=None, sideband=None, dtype=None):
        self.dechannelized = Dechannelize(
            ih, n=n, samples_per_frame=None, frequency=frequency,
            sideband=sideband, dtype=dtype)
        self._FFT = self.dechannelized._FFT
        n_tap, n = response.shape
        pad = (n_tap - 1) * n
        if samples_per_frame is not None:
            samples_per_frame = samples_per_frame + pad
        assert pad % 2 == 0
        super().__init__(self.dechannelized, pad_start=pad//2, pad_end=pad//2,
                         samples_per_frame=samples_per_frame)
        self._response = response
        self._reshape = ((self._ih_samples_per_frame // n, n)
                         + self.ih.sample_shape)
        self._ppf_fft = self._FFT(
            shape=self._reshape, dtype=self.dtype)
        self._ppf_ifft = self._ppf_fft.inverse()
        self._inv_sn2 = 1. / (sn * sn)

    @lazyproperty
    def _ft_inverse_response(self):
        long_response = np.zeros(self._reshape[:2], self.dtype)
        long_response[:self._response.shape[0]] = self._response
        long_response.shape = (long_response.shape
                               + (1,) * len(self.ih.sample_shape))
        fft = self._FFT(shape=long_response.shape, dtype=self.ih.dtype)
        ft_response = fft(long_response).conj()
        inverse = (ft_response.conj()
                   / (ft_response.real ** 2 + ft_response.imag ** 2
                      + self._inv_sn2))
        return inverse

    def task(self, data):
        data = data.reshape(self._reshape)
        ft = self._ppf_fft(data)
        ft *= self._ft_inverse_response
        result = self._ppf_ifft(ft)
        # Remove padding.
        result = result[self._response.shape[0]-1:]
        # Reshape as timestream
        return result.reshape((-1,) + result.shape[2:])
