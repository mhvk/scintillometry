import numpy as np


def sinc_window(ntap, lblock):
    """Sinc window function.

    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    coeff_length = np.pi * ntap
    coeff_num_samples = ntap * lblock

    # Sampling locations of sinc function
    X = np.arange(-coeff_length / 2.0, coeff_length / 2.0,
                  coeff_length / coeff_num_samples)

    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so use X/pi
    return np.sinc(X / np.pi)


def sinc_hamming(ntap, lblock):
    """Hamming-sinc window function.

    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """

    return sinc_window(ntap, lblock) * np.hamming(ntap * lblock)


def pfb(timestream, nfreq, ntap=4, window=sinc_hamming):
    """Perform the CHIME PFB on a timestream.

    Parameters
    ----------
    timestream : np.ndarray
        Timestream to process
    nfreq : int
        Number of frequencies we want out (probably should be odd
        number because of Nyquist)
    ntaps : int
        Number of taps.

    Returns
    -------
    pfb : np.ndarray[:, nfreq]
        Array of PFB frequencies.
    """

    # Number of samples in a sub block
    lblock = 2 * (nfreq - 1)

    # Number of blocks
    nblock = timestream.size // lblock - (ntap - 1)

    # Initialise array for spectrum
    spec = np.zeros((nblock, nfreq), dtype=np.complex128)

    # Window function
    w = window(ntap, lblock)

    # Iterate over blocks and perform the PFB
    for bi in range(nblock):
        # Cut out the correct timestream section
        ts_sec = timestream[(bi*lblock):((bi+ntap)*lblock)].copy()

        # Perform a real FFT (with applied window function)
        ft = np.fft.rfft(ts_sec * w)

        # Choose every n-th frequency
        spec[bi] = ft[::ntap]

    return spec
