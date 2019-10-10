# Licensed under the GPLv3 - see LICENSE
"""Tasks for integration over time and pulse phase."""

import operator
import warnings

import numpy as np
from scipy.sparse import csr_matrix
import astropy.units as u
from astropy.utils import ShapedLikeNDArray, lazyproperty

from .base import BaseTaskBase


__all__ = ['Integrate', 'Fold', 'Stack']


class _FakeOutput(ShapedLikeNDArray):
    """Pretend output class that diverts setting.

    It subclasses `~astropy.utils.ShapedLikeNDArray` to mimic all the
    shape properties of `~numpy.ndarray` given a (fake) shape.
    """
    def __init__(self, shape, setitem):
        self._shape = shape
        self._setitem = setitem

    def __setitem__(self, item, value):
        # Call back on out[item] = value.
        return self._setitem(item, value)

    # The two required parts for ShapedLikeNDArray.
    @property
    def shape(self):
        return self._shape

    def _apply(self, *args, **kwargs):
        raise NotImplementedError("No _apply possible for _FakeOutput")


def is_index(n):
    """Helper that checks whether n is suitable for indexing."""
    try:
        operator.index(n)
    except TypeError:
        return False
    else:
        return True


class Integrate(BaseTaskBase):
    """Integrate a stream stepwise.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    step : int or `~astropy.units.Quantity`, optional
        Interval over which to integrate.  For time invervals, should have
        units of time.  For phase intervals, units should be consistent with
        what the ``phase`` callable returns.  If no ``phase`` callable is
        passed in, then if step is integer, it is taken to be a number of
        samples in the underlying stream that should be integrated over,
        and if step is omitted, integration is over all samples.
    phase : callable
        Should return full pulse phase (i.e., including cycle count) for given
        input times (passed in as '~astropy.time.Time').  The output should be
        compatible with ``step``, i.e., generally an `~astropy.units.Quantity`
        with angular units.
    start : `~astropy.time.Time` or int, optional
        Time or offset at which to start the integration. If an offset or if
        ``step`` is integer, the actual start time will the underlying sample
        time nearest to the requested one.  Default: 0 (start of stream).
    average : bool, optional
        Whether the output should be the average of all entries that
        contributed to it, or rather the sum, in a structured array that holds
        both ``'data'`` and ``'count'`` items.
    samples_per_frame : int, optional
        Number of samples to process in one go.  This can be used to optimize
        the process.  With many samples per bin, the default of 1 should be OK.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    Notes
    -----

    Since time or phase bins are typically not an integer multiple of the
    underlying bin spacing, the integrated samples will generally not contain
    the same number of samples.  The actual number of samples is counted, and
    for ``average=True``, the sums have been divided by these counts, with bins
    with no points set to ``NaN``.  For ``average=False``, the arrays returned
    by ``read`` are structured arrays with ``data`` and ``count`` fields.

    .. warning: The format for ``average=False`` may change in the future.

    """
    def __init__(self, ih, step=None, phase=None, *,
                 start=0, average=True, samples_per_frame=1, dtype=None):
        ih_start = ih.seek(start)
        ih_n_sample = ih.shape[0] - ih_start
        if ih_start < 0 or ih_n_sample < 0:
            raise ValueError("'start' is not within the underlying stream.")

        if phase is None and (step is None or is_index(step)):
            # For integer step, avoid calculating time if possible,
            # since if a Stack was run before, start_time and stop_time
            # are no longer reliable.
            start_time = False
            if step is None:
                step = ih_n_sample

            sample_rate = ih.sample_rate / step
            shape = (ih_n_sample // step,) + ih.sample_shape
            # Initialize values for _get_offsets.
            self._mean_offset_size = 1. / step

        else:
            try:
                # We may not be at an integer sample.
                ih_start += ((start - ih.time) * ih.sample_rate).to_value(u.one)
            except TypeError:  # start is not a Time
                start = ih.time
            start_time = start
            stop = ih.stop_time
            if phase is not None:
                start = phase(start)
                stop = phase(stop)

            sample_rate = 1. / step
            n_sample = ((stop - start) / step).to_value(u.one)
            shape = (int(n_sample),) + ih.sample_shape
            # Initialize values for _get_offsets.
            self._mean_offset_size = n_sample / ih_n_sample
            self._start = start

        if dtype is None:
            if average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        super().__init__(ih, shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         start_time=start_time, dtype=dtype)
        self.average = average
        self._phase = phase
        self._ih_start = ih_start

    def _ih_time(self, offset):
        """Get time in underlying stream for given offset.

        Ensures the offset is put back, and adds a hopefully helpful
        note to any exception.
        """
        ih_offset = self.ih.tell()
        try:
            self.ih.seek(offset)
            return self.ih.time
        except Exception as exc:
            exc.args += ('cannot calculate time; this can happen if '
                         'integrating a pulse stack',)
            raise exc
        finally:
            self.ih.seek(ih_offset)

    @lazyproperty
    def start_time(self):
        """Start time of the output, left edge of the first sample.

        See also `time` and `stop_time`.
        """
        if self._start_time:
            return self._start_time
        else:
            return self._ih_time(self._ih_start)

    @lazyproperty
    def stop_time(self):
        """Time at the end of the output, just after the last sample.

        See also `start_time` and `time`.
        """
        raw_stop = self._get_offsets(self.shape[0])
        return self._ih_time(raw_stop)

    def _get_offsets(self, samples, precision=1.e-3, max_iter=10):
        """Get offsets in the underlying stream nearest to samples.

        For a phase callable, this is done by iteratively guessing offsets,
        calculating their associated phase, and updating, until the change
        in guessed offset is less than ``precision`` or more than ``max_iter``
        iterations are done.

        Phase is assumed to increase monotonously with time.
        """
        if self._phase is None:
            return (np.around(samples / self._mean_offset_size +
                              self._ih_start).astype(int))

        # Requested phases relative to start (we work relative to the start
        # to avoid rounding errors for large cycle counts).  Also, we want
        # *not* to use the Phase class, as it makes interpolation tricky.
        phase = np.ravel(samples) / self.sample_rate
        # Initial guesses for the associated offsets.
        ih_mean_phase_size = self._mean_offset_size / self.sample_rate
        offsets = (phase / ih_mean_phase_size).to_value(u.one)
        # In order to update guesses, below we interpolate phase in offset.
        # Add known boundaries to ensure we do not go out of bounds there.
        all_offsets = np.hstack((0, offsets, self.ih.shape[0] - self._ih_start))
        # Associated phases relative to start phase;
        # all but start (=0) and stop will be overwritten.
        all_ih_phase = all_offsets * ih_mean_phase_size
        # Add in base offset in underlying file.
        all_offsets += self._ih_start
        # Select the parts we are going to modify (in-place).
        offsets = all_offsets[1:-1]
        ih_phase = all_ih_phase[1:-1]
        mask = np.ones(offsets.shape, bool)
        it = 0
        while np.any(mask) and it < max_iter:
            # Use mask to avoid calculating more phases than necessary.
            # First calculate phase associate with the current offset guesses.
            old_offsets = offsets[mask]
            ih_time = self.ih.start_time + old_offsets / self.ih.sample_rate
            # TODO: the conversion is necessary because Quantity(Phase)
            # doesn't convert the two doubles to float internally.
            ih_phase[mask] = (self._phase(ih_time) -
                              self._start).astype(ih_phase.dtype, copy=False)
            # Next, interpolate in known phases to get improved offsets.
            offsets[mask] = np.interp(phase[mask], all_ih_phase, all_offsets)
            # Finally, update mask.
            mask[mask] = abs(offsets[mask] - old_offsets) > precision
            it += 1

        if it >= max_iter:  # pragma: no cover
            warnings.warn('offset calculation did not converge. '
                          'This should not happen!')

        shape = getattr(samples, 'shape', ())
        return offsets.round().astype(int).reshape(shape)

    def _read_frame(self, frame_index):
        """Determine which samples to read, and integrate over them.

        Uses the a ``_get_offsets`` method to determine where in the underlying
        stream the samples should be gotten from.

        Integration is done by setting up a fake output array whose setter
        calls back to the ``_integrate`` method that does the actual summing.
        """
        # Get offsets in the underlying stream for the current samples (and
        # the next one to get the upper edge). For integration over time
        # intervals, these offsets are not necessarily evenly spaced.
        samples = (frame_index * self.samples_per_frame +
                   np.arange(self.samples_per_frame + 1))
        offsets = self._get_offsets(samples)
        self.ih.seek(offsets[0])
        offsets -= offsets[0]
        # Set up fake output with a shape that tells the reader of the
        # underlying stream how many samples should be read (and a remaining
        # part that should pass consistency checks), and which has a callback
        # for the actual setting of output in the reader.
        integrating_out = _FakeOutput((offsets[-1],) + self.ih.sample_shape,
                                      setitem=self._integrate)
        # Set up real output and store information used in self._integrate
        frame = np.zeros((self.samples_per_frame,) + self.sample_shape,
                         dtype=self.dtype)
        if self.average:
            ndim_ih_sample = len(self.ih.sample_shape)
            self._frame = {
                'data': frame,
                'count': np.zeros(frame.shape[:-ndim_ih_sample] +
                                  (1,) * ndim_ih_sample, dtype=int)}
        else:
            self._frame = frame
        self._offsets = offsets

        # Do the actual reading.
        self.ih.read(out=integrating_out)
        if self.average:
            frame /= self._frame['count']

        return frame

    def _integrate(self, item, data):
        """Sum data in the correct samples.

        Here, item will be a slice with start and stop being indices in the
        underlying stream relative to the start of the current output frame,
        and data the corresponding slice of underlying stream.
        """
        # Note that this is not entirely trivial even for integrating over an
        # integer number of samples, since underlying data frames do not
        # necessarily contain integer multiples of this number of samples.
        #
        # First find all samples that have any overlap with the slice, i.e.,
        # for which start < offset_right and stop > offset_left.  Here, we use
        # the offsets in the underlying stream for each sample in the current
        # frame, plus the one just above, i.e., f[0]=0, f[1], f[2], ..., f[n].
        # (E.g., bin 0 should be included only when start < f[1]; bin n-1
        # only when stop > f[n-1].)
        start = np.searchsorted(self._offsets[1:], item.start, side='right')
        stop = np.searchsorted(self._offsets[:-1], item.stop, side='left')
        # Calculate corresponding indices in ``data`` by extracting the offsets
        # that have any overlap (we take one more -- guaranteed to exist --
        # so that we can count the number of items more easily), subtracting
        # the start offset, and clipping to the right range.
        indices = self._offsets[start:stop + 1] - item.start
        indices[0] = 0
        indices[-1] = item.stop - item.start
        # Finally, sum within slices constructed from consecutive indices
        # (reduceat always adds the end point itself).
        self._frame['data'][start:stop] += np.add.reduceat(data, indices[:-1])
        self._frame['count'][start:stop] += (
            np.diff(indices).reshape((-1,) + (1,) * (data.ndim - 1)))


class Fold(Integrate):
    """Fold pulse profiles in fixed time intervals.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_phase : int
        Number of bins per pulse period.
    phase : callable
        Should return pulse phases (with or without cycle count) for given
        input time(s), passed in as an '~astropy.time.Time' object.  The output
        can be an `~astropy.units.Quantity` with angular units or a regular
        array of float (in which case units of cycles are assumed).
    step : int or `~astropy.units.Quantity`, optional
        Number of input samples or time interval over which to fold.
        If not given, the whole file will be folded into a single profile.
    start : `~astropy.time.Time` or int, optional
        Time or offset at which to start the integration. If an offset or if
        ``step`` is integer, the actual start time will the underlying sample
        time nearest to the requested one.  Default: 0 (start of stream).
    average : bool, optional
        Whether the output pulse profile should be the average of all entries
        that contributed to it, or rather the sum, in a structured array that
        holds both ``'data'`` and ``'count'`` items.
    samples_per_frame : int, optional
        Number of sample times to process in one go.  This can be used to
        optimize the process, though in general the default of 1 should work.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    See Also
    --------
    Stack : to integrate over pulse phase and create pulse stacks

    Notes
    -----
    Since the sample time is not necessarily an integer multiple of the pulse
    period, the returned profiles will generally not contain the same number
    of samples in each phase bin.  The actual number of samples is counted,
    and for ``average=True``, the sums have been divided by these counts, with
    bins with no points set to ``NaN``.  For ``average=False``, the arrays
    returned by ``read`` are structured arrays with ``data`` and ``count``
    fields.

    .. warning: The format for ``average=False`` may change in the future.

    """
    def __init__(self, ih, n_phase, phase, step=None, *,
                 start=0, average=True, samples_per_frame=1, dtype=None):
        super().__init__(ih, step=step, start=start, average=average,
                         samples_per_frame=samples_per_frame)
        # And ensure we reshape it to cycles.
        self._shape = (self._shape[0], n_phase) + ih.sample_shape
        self.n_phase = n_phase
        self.phase = phase

    def _read_frame(self, frame_index):
        # Before calling the underlying implementation, get the start time in
        # the underlying frame, to be used to calculate phases in _integrate.
        offset0 = self._get_offsets(frame_index * self.samples_per_frame)
        self.ih.seek(offset0)
        self._raw_time = self.ih.time
        return super()._read_frame(frame_index)

    def _integrate(self, item, raw):
        # Get sample and phase indices.
        raw_items = np.arange(item.start, item.stop)
        # map raw sample to folded sample.
        sample_map = {}
        if self.samples_per_frame == 1:
            sample_index = 0
            sample_map[sample_index] = (0, len(raw_items))
        else:
            sample_index = np.searchsorted(self._offsets[1:], raw_items)
            # Detect where the sample index changes
            sample_index_diff = np.diff(sample_index)
            edges = np.where(sample_index_diff != 0)[0] + 1
            pad_edges = np.hstack((edges, len(raw_items)))
            sample_map[sample_index[0]] = (0, pad_edges[0])
            for ii, e in enumerate(edges):
                sample_map[sample_index[e]] = (e, pad_edges[ii + 1])


        # TODO: allow having a phase reference.
        phases = self.phase(self._raw_time + raw_items / self.ih.sample_rate)
        phase_index = ((phases % (1. * u.cycle)).to_value(u.cycle) *
                       self.n_phase).astype(int)

        for sp_idx, raw_sp_idx in sample_map.items():
            target_phase_index = phase_index[raw_sp_idx[0]:raw_sp_idx[1]]
            n_ph_sample = target_phase_index.shape[0]
            index_map_matrix = csr_matrix((np.ones(n_ph_sample),
                                          (target_phase_index,
                                           np.arange(n_ph_sample))),
                                           shape=(self.n_phase, n_ph_sample)).toarray()
            self._frame['data'][sp_idx] += np.tensordot(index_map_matrix,
                                               raw[raw_sp_idx[0]:raw_sp_idx[1]],
                                               axes=([1,], [0,]))
            counts = np.sum(index_map_matrix, axis=1,
                            dtype=int).reshape((self.sample_shape[0],) +
                                               (1,) * len(self.sample_shape[1:]))
            self._frame['count'][sp_idx] += counts


class Stack(BaseTaskBase):
    """Create a stream of pulse profiles.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_phase : int
        Number of bins per pulse period.
    phase : callable
        Should return pulse phases for given input time(s), passed in as an
        '~astropy.time.Time' object.  The output should be an array of float,
        and has to include the cycle count.
    start : `~astropy.time.Time` or int, optional
        Time or offset at which to start the integration. If an offset or if
        ``step`` is integer, the actual start time will the underlying sample
        time nearest to the requested one.  Default: 0 (start of stream).
    average : bool, optional
        Whether the output pulse profile should be the average of all entries
        that contributed to it, or rather the sum, in a structured array that
        holds both ``'data'`` and ``'count'`` items.
    samples_per_frame : int, optional
        Number of sample times to process in one go.  This can be used to
        optimize the process, though with many samples per pulse period the
        default of 1 should be fine.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    See Also
    --------
    Fold : to calculate pulse profiles integrated over a given amount of time.

    Notes
    -----
    One can follow this with a `~scintillometry.integration.Integrate` task
    to average over multiple pulses.

    Since phase bins are typically not an integer multiple of the underlying
    bin spacing, the integrated samples will generally not contain the same
    number of samples.  The actual number of samples is counted, and for
    ``average=True``, the sums have been divided by these counts, with bins
    with no points set to ``NaN``.  For ``average=False``, the arrays returned
    by ``read`` are structured arrays with ``data`` and ``count`` fields.

    .. warning: The format for ``average=False`` may change in the future.

    """
    def __init__(self, ih, n_phase, phase, *,
                 start=0, average=True, samples_per_frame=1, dtype=None):
        # Set up the integration in phase bins.
        phased = Integrate(ih, u.cycle/n_phase, phase,
                           start=start, average=average,
                           samples_per_frame=samples_per_frame*n_phase,
                           dtype=dtype)
        # And ensure we reshape it to cycles.
        shape = (phased.shape[0] // n_phase, n_phase) + phased.shape[1:]
        super().__init__(phased, shape=shape,
                         sample_rate=phased.sample_rate / n_phase,
                         samples_per_frame=samples_per_frame,
                         dtype=dtype)
        self.n_phase = n_phase

    def _read_frame(self, frame_index):
        # Read frame in phased directly, bypassing its ``read`` method.
        out = self.ih._read_frame(frame_index)
        return out.reshape((self.samples_per_frame,) + self.sample_shape)

    @property
    def stop_time(self):
        """Time at the end of the output, just after the last sample."""
        return self.ih.stop_time
