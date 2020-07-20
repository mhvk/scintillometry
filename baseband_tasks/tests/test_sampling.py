#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test resampling."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time

from ..sampling import Resample, float_offset, TimeShift
from ..base import Task, SetAttribute
from ..combining import Stack
from ..generators import StreamGenerator


class Cosine:
    def __init__(self, frequency, start_time):
        self.frequency = frequency
        self.start_time = start_time

    def __call__(self, ih):
        dt = ((ih.time - self.start_time).to(u.s) +
              np.arange(ih.samples_per_frame) / ih.sample_rate)
        dt = dt.reshape((-1,) + (1,) * self.frequency.ndim)
        phi = (self.frequency * dt * u.cycle).to_value(u.rad)
        if ih.dtype.kind == 'f':
            cosine = np.cos(phi)
        else:
            cosine = np.exp(1j * phi)
        return cosine.astype(ih.dtype, copy=False)


class TestResampleReal:

    dtype = 'f4'
    atol = 1e-4
    sample_rate = 1 * u.kHz
    samples_per_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    frequency = 400. * u.kHz
    shape = (2048, 2)
    sideband = np.array([-1, 1])

    def setup(self):
        f_sine = self.sample_rate / 32 * np.ones(self.shape[1:])

        cosine = Cosine(f_sine, self.start_time)

        self.full_fh = StreamGenerator(
            cosine, shape=self.shape,
            sample_rate=self.sample_rate,
            samples_per_frame=self.samples_per_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.part_fh = StreamGenerator(
            cosine, shape=(self.shape[0] // 4,) + self.shape[1:],
            sample_rate=self.sample_rate / 4,
            samples_per_frame=self.samples_per_frame // 4,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

    def test_setup(self):
        full = self.full_fh.read()
        part = self.part_fh.read()
        assert np.all(part == full[::4])

    @pytest.mark.parametrize('offset',
                             (0., 0.25, 0.5, 1., 1.75, 10.5,
                              10.*u.ms, 0.015*u.s,
                              Time('2010-11-12T13:14:15.013')))
    def test_resample(self, offset):
        ih = Resample(self.part_fh, offset, samples_per_frame=512)
        # Always lose 1 sample per frame.
        assert ih.shape == ((self.part_fh.shape[0] - 1,)
                            + self.part_fh.sample_shape)
        # Check we are at the given offset.
        if isinstance(offset, Time):
            expected_time = offset
        elif isinstance(offset, u.Quantity):
            expected_time = self.part_fh.start_time + offset
        else:
            expected_time = (self.part_fh.start_time
                             + offset / self.part_fh.sample_rate)
        assert abs(ih.time - expected_time) < 1. * u.ns

        ioffset, fraction = divmod(float_offset(self.part_fh, offset), 1)
        assert ih.offset == ioffset
        expected_start_time = (self.part_fh.start_time
                               + fraction / self.part_fh.sample_rate)
        assert abs(ih.start_time - expected_start_time) < 1. * u.ns
        ih.seek(0)
        data = ih.read()
        expected = self.full_fh.read()[int(fraction*4):-(4-int(fraction*4)):4]
        assert_allclose(data, expected, atol=self.atol, rtol=0)


class TestResampleComplex(TestResampleReal):

    dtype = 'c16'
    atol = 1e-8


class StreamArray(StreamGenerator):
    def __init__(self, data, *args, **kwargs):
        def from_data(handle):
            return data[handle.offset:
                        handle.offset+handle.samples_per_frame]
        super().__init__(from_data, *args, **kwargs)


class TestResampleNoise(TestResampleComplex):

    dtype = 'c8'
    atol = 1e-4

    def setup(self):
        # Make noise with only frequencies covered by part.
        part_ft_noise = (np.random.normal(size=512*2*2)
                         .view('c16').reshape(-1, 2))
        # Make corresponding FT for full frame.
        full_ft_noise = np.concatenate((part_ft_noise[:256],
                                        np.zeros((512*3, 2), 'c16'),
                                        part_ft_noise[-256:]), axis=0)
        part_data = np.fft.ifft(part_ft_noise, axis=0)
        # Factor 2048/512 to ensure data have same power.
        full_data = np.fft.ifft(full_ft_noise * 2048 / 512, axis=0)

        self.full_fh = StreamArray(
            full_data, shape=self.shape,
            sample_rate=self.sample_rate,
            samples_per_frame=self.samples_per_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.part_fh = StreamArray(
            part_data, shape=(self.shape[0] // 4,) + self.shape[1:],
            sample_rate=self.sample_rate / 4,
            samples_per_frame=self.samples_per_frame // 4,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)


def mix_downsample8(ih, data):
    if ih.complex_data:
        return (data[:, 0] * data[:, 1].conj())[::8]
    else:
        return (data[:, 0] * data[:, 1])[::8]


class TestTimeShift:
    dtype = np.dtype('c8')
    full_sample_rate = 102.4 * u.kHz
    sample_rate = full_sample_rate / 8
    samples_per_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    frequency = full_sample_rate * 7 / 8
    shape = (102400, 2)
    sideband = np.array([-1, 1])
    f_sine = frequency + sample_rate / 16 * sideband
    delay = 128
    delay_time = delay / full_sample_rate

    def setup(self):
        cosine = Cosine(self.f_sine, self.start_time)
        mixer = Cosine(self.frequency * self.sideband, self.start_time)

        self.full_fh = StreamGenerator(
            cosine, shape=self.shape,
            sample_rate=self.full_sample_rate,
            samples_per_frame=self.samples_per_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.mixer_fh = StreamGenerator(
            mixer, shape=self.shape,
            sample_rate=self.full_sample_rate,
            samples_per_frame=self.samples_per_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.tel1 = Task(Stack((self.full_fh, self.mixer_fh), axis=1),
                         mix_downsample8, sample_rate=self.sample_rate,
                         shape=(self.shape[0] // 8, 2))

        self.delayed_fh = SetAttribute(
            self.full_fh, start_time=self.start_time-self.delay_time)
        delayed_mix = Stack((self.delayed_fh, self.mixer_fh), axis=1,
                            samples_per_frame=128)
        self.tel2 = Task(delayed_mix, mix_downsample8,
                         sample_rate=self.sample_rate,
                         shape=(delayed_mix.shape[0] // 8, 2))

    def test_setup(self):
        data1 = self.tel1.read()
        dt1 = np.arange(self.tel1.shape[0]) / self.tel1.sample_rate
        phi1 = dt1[:, np.newaxis] * u.cycle * (self.f_sine - self.frequency)
        expected1 = np.exp(phi1.to_value(u.radian) * 1j)
        assert_allclose(data1, expected1, atol=1e-4, rtol=0)
        data2 = self.tel2.read()
        assert data2.shape[0] == (self.shape[0] - self.delay) // 8
        dt2 = dt1[self.delay//8:]
        phi2 = dt2[:, np.newaxis] * u.cycle * (self.f_sine - self.frequency)
        phi2 += self.delay / self.full_sample_rate * self.frequency * u.cycle
        expected2 = np.exp(phi2.to_value(u.radian) * 1j)
        assert_allclose(data2, expected2, atol=1e-4, rtol=0)
