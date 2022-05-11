"""Functionality to transform the audio input in the same way
that the Quartznet model expects it.
"""

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021-2022 scart97

__all__ = [
    "FeatureBatchNormalizer",
    "DitherAudio",
    "PreEmphasisFilter",
    "PowerSpectrum",
    "MelScale",
    "FilterbankFeatures",
]

import math
from typing import Optional, Tuple

import torch
from torch import nn

from fastcore.all import Transform
from fastaudio.all import AudioTensor


def normalize_tensor(
    input_values,
    mask: Optional[torch.Tensor] = None,
    div_guard: float = 1e-7,
    dim: int = -1,
):
    """Normalize tensor values, optionally using some mask to define the valid region.
    Args:
        input_values: input tensor to be normalized
        mask: Optional mask describing the valid elements.
        div_guard: value used to prevent division by zero when normalizing.
        dim: dimension used to calculate the mean and variance.
    Returns:
        Normalized tensor
    """

    mean = input_values.mean(dim=dim, keepdim=True).detach()
    std = (input_values.var(dim=dim, keepdim=True).detach() + div_guard).sqrt()
    return (input_values - mean) / std


class FeatureBatchNormalizer(nn.Module):
    def __init__(self):
        """Normalize batch at the feature dimension."""
        super().__init__()
        self.div_guard = 1e-5

    def forward(
        self, x
    ):
        """
        Args:
            x: Tensor of shape (batch, features, time)
        """
        # https://github.com/pytorch/pytorch/issues/45208
        # https://github.com/pytorch/pytorch/issues/44768
        with torch.no_grad():
            return (
                normalize_tensor(x, div_guard=self.div_guard)
            )


class DitherAudio(nn.Module):
    def __init__(self, dither: float = 1e-5):
        """Add some dithering to the audio tensor.

        Note:
            From wikipedia: Dither is an intentionally applied
            form of noise used to randomize quantization error.

        Args:
            dither: Amount of dither to add.
        """
        super().__init__()
        self.dither = dither

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time)
        """
        if self.training:
            return x + (self.dither * torch.randn_like(x))
        else:
            return x


class PreEmphasisFilter(nn.Module):
    def __init__(self, preemph: float = 0.97):
        """Applies preemphasis filtering to the audio signal.
        This is a classic signal processing function to emphasise
        the high frequency portion of the content compared to the
        low frequency. It applies a FIR filter of the form:

        `y[n] = y[n] - preemph * y[n-1]`

        Args:
            preemph: Filter control factor.
        """
        super().__init__()
        self.preemph = preemph

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time)
        """
        return torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )


class PowerSpectrum(nn.Module):
    def __init__(
        self,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        n_fft: Optional[int] = None,
    ):
        """Calculates the power spectrum of the audio signal, following the same
        method as used in NEMO.

        Args:
            n_window_size: Number of elements in the window size.
            n_window_stride: Number of elements in the window stride.
            n_fft: Number of fourier features.

        Raises:
            ValueError: Raised when incompatible parameters are passed.
        """
        super().__init__()
        if n_window_size <= 0 or n_window_stride <= 0:
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        window_tensor = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window_tensor)
        # This way so that the torch.stft can be changed to the patched version
        # before scripting. That way it works correctly when the export option
        # doesnt support fft, like mobile or onnx.
        self.stft_func = torch.stft

    @torch.no_grad()
    def forward(
        self, x):
        """
        Args:
            x: Tensor of shape (batch, time)
        """
        x = self.stft_func(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(device = x.device, dtype=torch.float),
            return_complex=False,
        )

        # torch returns real, imag; so convert to magnitude
        x = torch.sqrt(x.pow(2).sum(-1))
        # get power spectrum
        x = x.pow(2.0)
        return x
    
def _create_triangular_filterbank(
    all_freqs,
    f_pts,
):
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb



def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels, mel_scale: str = "htk"):
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs

    
    
    
def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk"):
    r"""Create a frequency bin conversion matrix.

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * melscale_fbanks(A.size(-1), ...)``.

    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb
    
class MelScale(nn.Module):
    def __init__(
        self, sample_rate: int, n_fft: int, nfilt: int, log_scale: bool = True
    ):
        """Convert a spectrogram to Mel scale, following the default
        formula of librosa instead of the one used by torchaudio.
        Also converts to log scale.

        Args:
            sample_rate: Sampling rate of the signal
            n_fft: Number of fourier features
            nfilt: Number of output mel filters to use
            log_scale: Controls if the output should also be applied a log scale.
        """
        super().__init__()

        filterbanks = (
            melscale_fbanks(
                int(1 + n_fft // 2),
                n_mels=nfilt,
                sample_rate=sample_rate,
                f_min=0,
                f_max=sample_rate / 2,
                norm="slaney",
                mel_scale="slaney",
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )
        self.register_buffer("fb", filterbanks)
        self.log_scale = log_scale

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, features, time)
        """
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(device=x.device,dtype=x.dtype), x)
        # log features
        # We want to avoid taking the log of zero
        if self.log_scale:
            x = torch.log(x + 2 ** -24)
        return x


    
class FilterbankFeatures(Transform):
    
    def __init__(self,
    sample_rate: int = 16000,
    n_window_size: int = 320,
    n_window_stride: int = 160,
    n_fft: int = 512,
    preemph: float = 0.97,
    nfilt: int = 64,
    dither: float = 1e-5):
        r"""Creates the Filterbank features used in the Quartznet model.

        Args:
            sample_rate: Sampling rate of the signal.
            n_window_size: Number of elements in the window size.
            n_window_stride: Number of elements in the window stride.
            n_fft: Number of fourier features.
            preemph: Preemphasis filtering control factor.
            nfilt: Number of output mel filters to use.
            dither: Amount of dither to add.
        """
        
        self.transform = nn.Sequential(
        DitherAudio(dither=dither), PreEmphasisFilter(preemph=preemph),
        PowerSpectrum(
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_fft=n_fft,
        ),
        MelScale(sample_rate=sample_rate, n_fft=n_fft, nfilt=nfilt),
        FeatureBatchNormalizer(),
        )
        
    def encodes(self,x:AudioTensor):
        return self.transform(x)

# def FilterbankFeatures(
#     sample_rate: int = 16000,
#     n_window_size: int = 320,
#     n_window_stride: int = 160,
#     n_fft: int = 512,
#     preemph: float = 0.97,
#     nfilt: int = 64,
#     dither: float = 1e-5,
# ) -> nn.Module:
#     """Creates the Filterbank features used in the Quartznet model.

#     Args:
#         sample_rate: Sampling rate of the signal.
#         n_window_size: Number of elements in the window size.
#         n_window_stride: Number of elements in the window stride.
#         n_fft: Number of fourier features.
#         preemph: Preemphasis filtering control factor.
#         nfilt: Number of output mel filters to use.
#         dither: Amount of dither to add.
#     Returns:
#         Module that computes the features based on raw audio tensor.
#     """
#     return nn.Sequential(
#         DitherAudio(dither=dither), PreEmphasisFilter(preemph=preemph),
#         PowerSpectrum(
#             n_window_size=n_window_size,
#             n_window_stride=n_window_stride,
#             n_fft=n_fft,
#         ),
#         MelScale(sample_rate=sample_rate, n_fft=n_fft, nfilt=nfilt),
#         FeatureBatchNormalizer(),
#     )


# def patch_stft(filterbank: nn.Module) -> nn.Module:
#     """This function applies a patch to the FilterbankFeatures to use instead a convolution
#     layer based stft. That makes possible to export to onnx and use the scripted model
#     directly on arm cpu's, inside mobile applications.

#     Args:
#         filterbank: the FilterbankFeatures layer to be patched

#     Returns:
#         Layer with the stft operation patched.
#     """
#     filterbank[1].stft_func = convolution_stft
#     return filterbank
