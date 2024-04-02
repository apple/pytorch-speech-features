#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Callable


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1127.0 * np.log(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (np.exp(mel / 1127.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    meldelta = (highmel - lowmel) / (nfilt + 1)
    bins = np.zeros((nfilt, torch.div(nfft, 2, rounding_mode="floor") + 1))
    freqdelta = samplerate / nfft
    for i in range(nfilt):
        left_mel = lowmel + i * meldelta
        centre_mel = lowmel + (i + 1) * meldelta
        right_mel = lowmel + (i + 2) * meldelta
        for j in range(torch.div(nfft, 2, rounding_mode="floor") + 1):
            freq = freqdelta * j
            mel = hz2mel(freq)
            if left_mel < mel < right_mel:
                #                 print(f"{left_mel} {centre_mel} {right_mel} {mel}")
                if mel <= centre_mel:
                    bins[i][j] = (mel - left_mel) / (centre_mel - left_mel)
                else:
                    bins[i][j] = (right_mel - mel) / (right_mel - centre_mel)
    return bins


def preprocess_wav_pyt(sig, winlen, winstep):
    """
    This function prepares the input signal by truncating it as per the winlen and winstep
    :param sig: Tensor containing the input signal in float32 (accepts 1D and 2D tensors)
    :param winlen: Integer specifying the window length to be used for FFT
    :param winstep: Integer specifying the window step to be used for FFT
    :return: Tensor containing the truncated input signal in float32
    """
    extra_frames = (sig.shape[-1] - winlen) % winstep
    if extra_frames > 0:
        if sig.ndim == 1:
            return sig[: sig.shape[-1] - (int(extra_frames) - 1)]
        elif sig.ndim == 2:
            return sig[:, : sig.shape[-1] - (int(extra_frames) - 1)]
        else:
            assert False, "Unsupported ndim of signal. allowed 1 or 2."

    else:
        return sig


def comp_mel_filters_npy(
    mel_min=64, mel_max=8000, num_mels=40, sr=16000, nfft=512, mel_filter_path=None
):
    """
    This function determines mel filter coefficients
    It either takes a pre-determined set of coefficients from a given mel filter numpy file
    or computes mel-filters using the kaldi implementation in get_filterbanks
    :param mel_min: Minimum frequency for mel spectrum (Hz value)
    :param mel_max: Maximum frequency for mel spectrum (Hz value)
    :param num_mels: Number of mel filterbanks needed
    :param sr: Sample rate of ainput audio
    :param nfft: Number of FFT bins
    :param mel_filter_path: (optional) path of numpy array to load frozen mel filter coefficients
    :return: mel filter coefficients in numpy format, shape (nfft/2+1 x num_mels)
    """
    if mel_filter_path is not None:
        mel_filt = np.load(mel_filter_path)
    else:
        mel_filt = get_filterbanks(num_mels, nfft, sr, mel_min, mel_max)

    if mel_filt.shape[1] != (int(nfft / 2) + 1) or mel_filt.shape[0] != num_mels:
        print("Wrong dimensions of mel filter coefficients")
        raise ValueError("incorrect dimensions")
    return mel_filt.T.astype(np.float32)


class DC_Filter(nn.Module):
    """
    Helps to remove dc component fromwav using AvgPooling
    """

    def __init__(self, winlen: int = 400):
        super(DC_Filter, self).__init__()
        self.mean_filter = torch.nn.AvgPool1d(winlen, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor containing windows of specified window length
        :return: Tensor after removing the DC component from each window
        """
        signal_T = torch.transpose(x, 2, 1)
        means = self.mean_filter(signal_T)
        means_T = means.transpose(2, 1)
        windows = x - means_T
        return windows


class Wav2Frames(nn.Module):
    """
    Create windows from continuous time signal of window length and window step
    winlen: Window length
    winstep: Window step
    """

    def __init__(self, winlen: int = 400, winstep: int = 160) -> None:
        super(Wav2Frames, self).__init__()
        self.winstep = winstep
        self.window_filt_weights = torch.nn.parameter.Parameter(
            torch.eye(winlen, dtype=torch.float32).unsqueeze(1), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor containing continuous time signals (batch_size x time )
        :return: Tensor containing windows spliced by window length
        and window step (batch_size x time/winstep x winlen)
        """
        return F.conv1d(x, self.window_filt_weights, stride=self.winstep)


class PreEmphasis(nn.Module):
    """
    Pre-emphasize an audio signal with a first-order auto-regressive filter:

        y[k] = y[k] - k * y[k-1]
    """

    def __init__(self, k: float = 0.97, requires_grad: bool = False) -> None:
        super(PreEmphasis, self).__init__()
        pr_filt = torch.tensor([[-k], [1]], dtype=torch.float32)
        self.pr_filt = torch.nn.parameter.Parameter(
            pr_filt.unsqueeze(0).unsqueeze(0), requires_grad=requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor containing windows
                (batch_size x time x window length)
        :return: Tensor containing pre-emphasized windows
                (batch_size x time x window length)
        """
        windows = torch.cat([x[:, 0:1, :], x], 1)
        windows = windows.unsqueeze(1)
        return F.conv2d(windows, self.pr_filt, stride=1, padding=0).squeeze(1)


class Spectrogram(nn.Module):
    """
    This module produces spectrogram features for a given input tensor of wav
    """

    def __init__(
        self,
        winlen: int = 400,
        winstep: int = 160,
        premph_k: float = 0.97,
        winfunc: Callable = torch.hamming_window,
        add_noise: bool = False,
        remove_dc: bool = True,
        power: bool = True,
        do_log: bool = False,
        scale_spectrogram: float = 1.0,
        requires_grad: bool = False,
    ) -> None:
        super(Spectrogram, self).__init__()
        self.winlen = winlen
        self.nfft = torch.pow(
            2, torch.floor(torch.log2(torch.tensor(winlen).float())) + 1
        ).long()
        self.winstep = winstep
        self.fft_win_fn = winfunc
        self.add_noise = add_noise
        self.power = 2 if power else 1
        self.do_log = do_log
        self.scale_spectrogram = scale_spectrogram
        self.requires_grad = requires_grad
        self.window_maker = Wav2Frames(winlen, winstep)
        if remove_dc:
            self.dc_remover = DC_Filter(winlen)
        else:
            self.dc_remover = None
        if premph_k > 0:
            self.pre_emphasis = PreEmphasis(premph_k)
        else:
            self.pre_emphasis = None
        self.conv_filter = torch.nn.parameter.Parameter(
            self._get_conv_filt(), requires_grad=self.requires_grad
        )
        self.padding_filter = torch.nn.parameter.Parameter(
            self._comp_pad_filt(self.nfft, self.winlen).float(), requires_grad=False
        )

    def _get_conv_filt(self) -> torch.Tensor:
        nfilt = int(self.nfft / 2) + 1
        cos_filter = torch.zeros((nfilt, self.nfft))
        sin_filter = torch.zeros((nfilt, self.nfft))
        if self.fft_win_fn is not None:
            win_func = self.fft_win_fn(self.winlen)
        else:
            win_func = torch.ones(self.winlen)
        for i in range(nfilt):
            for j in range(self.winlen):
                cos_filt = torch.cos(2 * math.pi * i * j / self.nfft) * win_func[j]
                sin_filt = -torch.sin(2 * math.pi * i * j / self.nfft) * win_func[j]
                cos_filter[i][j] = cos_filt
                sin_filter[i][j] = sin_filt
        conv_filter = torch.cat((cos_filter, sin_filter), dim=0)
        conv_filter = conv_filter.float()
        conv_filter = conv_filter.unsqueeze(1).transpose(-1, -2)
        return conv_filter

    def _comp_pad_filt(self, nfft: int, winlen: int) -> torch.Tensor:
        """
        This function computes the padding filter to be used for convolution
        required to right pad windows before FFT
        :param nfft: num fft bins
        :param winlen: window length used for FFT
        :return: numpy array containinng the padding filter
        """
        fft_filt = torch.eye(winlen)
        zero_filt = torch.zeros((winlen, nfft - winlen))
        fft_filt = torch.cat((fft_filt, zero_filt), dim=1)
        return fft_filt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Time domain signal (B x T)
        :return: Spectral features (B x T/winstep x nfft/2+1)
        """
        # x = preprocess_wav_pyt(x, self.winlen, self.winstep)
        if self.add_noise:
            x += torch.randn(x.size(), device=x.device)
        x = x.unsqueeze(1)
        windows = self.window_maker(x)
        if self.dc_remover is not None:
            dc_removed_windows = self.dc_remover(windows)
        else:
            dc_removed_windows = windows
        if self.pre_emphasis is not None:
            preemph_windows = self.pre_emphasis(dc_removed_windows)
        else:
            preemph_windows = dc_removed_windows
        fft_windows = preemph_windows.transpose(-1, -2)
        fft_signal = torch.matmul(fft_windows, self.padding_filter)
        fft_signal = fft_signal.transpose(-1, -2)
        conv_out = F.conv1d(fft_signal, self.conv_filter, None, 1, 0)
        conv_out = conv_out.transpose(2, 1).unsqueeze(-1)
        conv_out = torch.cat(
            torch.split(conv_out, conv_out.shape[2] // 2, dim=2), dim=-1
        )
        mag_spec = torch.sqrt(conv_out.pow(2).sum(dim=3)).pow(self.power)
        if self.do_log:
            return torch.log(
                self.scale_spectrogram * torch.clamp(mag_spec, min=1.16549e-38)
            )
        return self.scale_spectrogram * mag_spec


class FBank(nn.Module):
    """
    Compute a mel-scaled spectrogram.
    """

    def __init__(
        self,
        sr: int = 16000,
        winlen: int = 400,
        winstep: int = 160,
        mel_filt_path: str = None,
        mel_min: float = 64,
        mel_max: float = 8000,
        num_mels: int = 40,
        premph_k: float = 0.97,
        winfunc: Callable = np.hamming,
        remove_dc: bool = True,
        add_noise: bool = False,
        do_log: bool = True,
        scale_spectrogram: float = 1.0,
        scale_fbanks: float = 1.0,
        requires_grad: bool = False,
    ) -> None:
        super(FBank, self).__init__()
        self.do_log = do_log
        self.scale_fbanks = scale_fbanks
        self.spectrogram_extractor = Spectrogram(
            winlen,
            winstep,
            premph_k,
            winfunc,
            add_noise,
            remove_dc,
            True,
            False,
            scale_spectrogram,
            requires_grad,
        )
        mel_filter = comp_mel_filters_npy(
            mel_min,
            mel_max,
            num_mels,
            sr,
            self.spectrogram_extractor.nfft,
            mel_filt_path,
        )

        self.mel_filter = torch.nn.parameter.Parameter(
            torch.tensor(mel_filter, requires_grad=requires_grad),
            requires_grad=requires_grad,
        )
        mask = np.zeros_like(mel_filter)
        mask[np.where(mel_filter != 0)] = 1
        self.mask = torch.nn.parameter.Parameter(
            torch.tensor(mask, requires_grad=requires_grad),
            requires_grad=requires_grad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input wav tensor (time domain) (B x T )
        :return: Output mel filterbank features (B x T/winstep x num_mels)
        """
        spectrogram = self.spectrogram_extractor(x)

        fbank_feats = torch.matmul(
            spectrogram,
            torch.nn.functional.relu(self.mel_filter * self.mask, inplace=True),
        )
        if self.do_log:
            return self.scale_fbanks * torch.log(
                torch.clamp(fbank_feats, min=1.16549e-38)
            )
        return fbank_feats


class MFCC(nn.Module):
    """
    Compute a mel-scaled spectrogram.
    """

    def __init__(
        self,
        sr: int = 16000,
        winlen: int = 400,
        winstep: int = 160,
        mel_filt_path: str = None,
        mel_min: float = 64,
        mel_max: float = 8000,
        num_mels: int = 40,
        premph_k: float = 0.97,
        winfunc: Callable = np.hamming,
        remove_dc: bool = True,
        add_noise: bool = False,
        num_mfcc: int = 26,
        do_log: bool = True,
        scale_spectrogram: float = 1.0,
        scale_fbanks: float = 1.0,
        requires_grad: bool = False,
    ) -> None:
        super(MFCC, self).__init__()
        self.do_log = do_log
        self.scale_fbanks = scale_fbanks
        self.fbank = FBank(
            sr,
            winlen,
            winstep,
            mel_filt_path,
            mel_min,
            mel_max,
            num_mels,
            premph_k,
            winfunc,
            remove_dc,
            add_noise,
            do_log,
            scale_spectrogram,
            scale_fbanks,
            requires_grad,
        )

        dct_filter = torch.tensor(
            self.comp_dct_filters_npy(None, num_mfcc, num_mels),
            requires_grad=requires_grad,
        )
        self.dct_filter = torch.nn.parameter.Parameter(
            dct_filter, requires_grad=requires_grad
        )
        self.roll_filter = torch.eye(num_mfcc)
        self.roll_filter = torch.nn.parameter.Parameter(
            torch.roll(self.roll_filter, dims=-1, shifts=-1), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Time domain signal (B x T)
        :return: Spectral features (B x T/winstep x nmfcc)
        """
        x = self.fbank(x)
        out = torch.matmul(x, self.dct_filter).matmul(self.roll_filter)
        return out

    def comp_dct_filters_npy(
        self, dct_filter_path: str = None, num_chans: int = None, num_mels: int = None
    ) -> np.ndarray:
        """
        This function determines the dct coefficients
        It either takes a pre-determined set of coefficients from a given dct filter numpy file
        or computes them on the fly using type 3 HTK dct implementation in _get_dct_filter
        :param dct_filter_path(optional): Path to predined dct filters
        :param num_chans: number of dct outputs
        :param num_mels: number of mel inputs
        :return:
        """
        if dct_filter_path is not None:
            dct_filt = np.load(dct_filter_path)
        else:
            assert num_chans is not None and num_mels is not None
            dct_filt = np.zeros((num_chans, num_mels))
            factor = np.sqrt(2.0 / num_mels)
            pIN = np.pi / num_mels
            for i in range(num_chans):
                for j in range(num_mels):
                    dct_filt[i][j] = factor * np.cos(pIN * i * (j + 0.5))

        return dct_filt.T.astype(np.float32)


class SlidingCMVN(nn.Module):
    """
    Calculates multiple moving average estiamtes given a kernel_size
    Similar to kaldi's apply-cmvn-sliding
    """

    def __init__(self, cmn_window: int = 600, min_cmn_window: int = 100) -> None:
        super(SlidingCMVN, self).__init__()
        self.cmn_window = cmn_window
        self.min_cmn_window = min_cmn_window
        self.padder = nn.ConstantPad1d((cmn_window - 1, 0), 0)
        self.avgpool1d = nn.AvgPool1d(kernel_size=cmn_window, padding=0, stride=1)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: torch.Tensor = None,
        mask: torch.TensorType = None,
    ) -> torch.Tensor:
        """
        :param x: Input tensor (BxTxD) / (TxD)
        :param seq_len(optional): Length of individual inputs in a batch
        :param mask(optional): Mask containing 1s till length of individual inputs in a batch
                    and 0 thereafter
        :return: Tensor (BxTxD) after applying sliding cmvn
        """
        self.device = x.device

        assert x.ndim in [2, 3], "Input needs to be tensor of shape B x T x D"
        if x.ndim == 2:
            x = x.unsqueeze(0)
            if seq_len is None:
                seq_len = torch.tensor(
                    x.shape[1], dtype=torch.long, device=self.device
                ).unsqueeze(0)
            if mask is None:
                mask = torch.ones((1, x.shape[1]), dtype=torch.long, device=self.device)
        else:
            assert seq_len is not None, "seq_len (Bx1) tensor not defined"
            assert mask is not None, "mask (BxT) is not defined"
        max_len = mask.shape[-1]
        # calculate mean1 used when feat_len < min_cmn_window
        temp_mask1 = torch.zeros(max_len, device=self.device)
        temp_mask1[: min(self.min_cmn_window, max_len)] = 1
        mean1_mask = (mask * temp_mask1).unsqueeze(-1)
        mean1_batch = x * mean1_mask
        sum1_batch = mean1_batch.sum(dim=1, keepdims=True)
        len1 = torch.hstack(
            (
                seq_len.unsqueeze(-1),
                self.min_cmn_window
                * torch.ones(seq_len.shape, device=self.device).unsqueeze(-1),
            )
        )
        mean1 = sum1_batch / torch.min(len1, dim=1)[0].unsqueeze(-1).unsqueeze(-1)
        mean1_batch -= mean1
        mean1_batch *= mean1_mask
        if torch.any(torch.isnan(mean1_batch)):
            raise ValueError
        # calculate mean2 used when min_cmn_window < feat_len < cmn_window
        temp_mask2 = torch.zeros(max_len, device=self.device)
        temp_mask2[
            min(self.min_cmn_window, max_len) : min(self.cmn_window, max_len)
        ] = 1
        mean2_mask = (mask * temp_mask2).unsqueeze(-1)
        len2 = torch.cumsum(mask, dim=-1)
        sum2_batch = x.cumsum(dim=1)
        mean2 = sum2_batch / len2.unsqueeze(-1)
        mean2_batch = x - mean2
        mean2_batch *= mean2_mask
        if torch.any(torch.isnan(mean2_batch)):
            raise ValueError
        # calculate mean3 used when feat_len > cmn_window
        temp_mask3 = torch.zeros(max_len, device=self.device)
        temp_mask3[min(max_len, self.cmn_window) : max_len] = 1
        mean3_mask = (mask * temp_mask3).unsqueeze(-1)
        padded_batch = self.padder(x.transpose(1, 2))
        avg_batch = self.avgpool1d(padded_batch).transpose(1, 2)
        mean3_batch = x - avg_batch
        mean3_batch *= mean3_mask
        if torch.any(torch.isnan(mean2_batch)):
            raise ValueError
        final_out = mean1_batch + mean2_batch + mean3_batch
        return final_out


class JoinAndSubsample(nn.Module):
    """
    Computes Splicing or SubSampling of input features
    """

    def __init__(
        self,
        leftpad: int = 0,
        rightpad: int = 0,
        stride: int = 1,
        feat_dim: int = 40,
        requires_grad: bool = False,
    ) -> None:
        super(JoinAndSubsample, self).__init__()
        num_ts = leftpad + rightpad + 1
        num_bins = feat_dim
        out_dim = feat_dim * num_ts
        conv_filter_npy = torch.zeros((out_dim, num_ts, num_bins))
        for i in range(out_dim):
            conv_filter_npy[i][int(i / num_bins)][i % num_bins] = 1
        conv_filter_ft = conv_filter_npy.float()
        conv_filter_ft = conv_filter_ft.unsqueeze(1)
        conv_filter_ft = torch.nn.parameter.Parameter(
            conv_filter_ft, requires_grad=requires_grad
        )
        self.conv_filter_ft = nn.Conv2d(
            out_dim,
            1,
            kernel_size=(num_ts, num_bins),
            stride=stride,
            padding=0,
            bias=False,
        )
        self.conv_filter_ft.weight.data = conv_filter_ft
        self.padding_ft = (0, 0, leftpad, rightpad)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor (B x T x D)
        :return: Tensor (B x T/stride x D*(leftpad + rightpad + 1))
        """

        self.conv_filter_ft = self.conv_filter_ft.to(x.device)

        assert x.ndim in [2, 3], "Input needs to be tensor of shape B x T x D or TxD"
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        padded_feats = F.pad(x, self.padding_ft, mode="replicate")
        torch_out = self.conv_filter_ft(padded_feats)
        return torch_out.squeeze(-1).transpose(1, 2)


class GlobalCMVN(nn.Module):
    """
    Applies a global normalization on inputs using cmvn_mean and cmvn_std
    """

    def __init__(
        self, cmvn_mean: np.ndarray, cmvn_std: np.ndarray, requires_grad: bool = False
    ) -> None:
        super(GlobalCMVN, self).__init__()
        assert cmvn_mean.ndim == 1 and cmvn_std.ndim == 1

        self.cmvn_mean = nn.Parameter(
            torch.tensor(cmvn_mean, dtype=torch.float32, requires_grad=requires_grad)
        )
        self.cmvn_std = nn.Parameter(
            torch.tensor(cmvn_std, dtype=torch.float32, requires_grad=requires_grad)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor (B x T x D)
        :return: Tensor (B x T x D)
        """
        assert x.ndim in [2, 3], (
            "GlobalCMVN Error : " "Input needs to be tensor of shape B x T x D or TxD"
        )
        assert (
            x.shape[-1] == self.cmvn_mean.shape[-1]
            and x.shape[-1] == self.cmvn_std.shape[-1]
        ), "CMVN stats not same shape at dimension -1 as input"
        self.to(x.device)
        return (x - self.cmvn_mean) / self.cmvn_std
