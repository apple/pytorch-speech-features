# PyTorch Speech Features

PyTorch based Feature Extraction to mimic [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi) feature extraction.

## Installation
``` bash
git clone https://github.com/apple/pytorch-speech-features.git
cd pytorch-speech-features
pip install .
```

## Usage
```python
import torch
from apple_pytorch_speech_features import FBank
feature_extractor = FBank()
dummy_wav = torch.randint(2,1000,(1,16000)).float() # create random integer tensor of shape (2,16000)
features = feature_extractor(dummy_wav) # outputs features of shape (2,98,40)
```

## Features
1. [Spectrogram](apple_pytorch_speech_features/feature_extraction.py#180): Spectrogram Extraction from time domain audio = Window --> Remove DC --> Pre-Emphasis --> STFT(DTFT) --> Power/Energy
    ```python
    sr=16000, # (Sample rate of input wav signal)
    winlen=400, # (Window length used for FFT/DFT/STFT)
    winstep=160, # (Window step used for FFT/DFT/STFT)
    premph_k=0.97, # (Pre-emphasis coefficient)
    winfunc=np.hamming, # (window function to be used for FFT)
    remove_dc=True, # (Remove mean after windowing)
    add_noise=True, # (same as dither in Kaldi)
    do_log=False, # (Produce results in log)
    scale_spectrogram=1, # (Add any constant scale factor to be used on spectral outputs (like 1/NFFT))
    requires_grad=False, # (Make dtft, pre-emphasis trainable if True)
    ```
2. [FBank](apple_pytorch_speech_features/feature_extraction.py#292): Mel-Filterbank Extraction = Spectrogram --> Mel-transform
   ```python
    sr=16000, # (Sample rate of input wav signal)
    winlen=400, # (Window length used for FFT/DFT/STFT)
    winstep=160, # (Window step used for FFT/DFT/STFT)
    mel_filt_path=None, # (Path of pre-calculated mel filter coefficients)
    mel_min=64, # (minimum frequency (Hz) for mel filter coefficient calculation)
    mel_max=8000, # (maximum frequency (Hz) for mel filter coefficient calculation)
    num_mels=40, # (Number of mel filterbanks)
    premph_k=0.97, # (Pre-emphasis coefficient)
    winfunc=np.hamming, # (window function to be used for FFT)
    remove_dc=True, # (Remove mean after windowing)
    add_noise=True, # (same as dither in Kaldi)
    do_log=True, # (Produce results in log)
    scale_spectrogram=1, # (Add any constant scale factor to be used on spectral outputs (like 1/NFFT))
    scale_fbanks=1, # (Add any constant scale factor to be used on fbank outputs like use 1/ln(10) if you want outputs in log 10 scale))
    requires_grad=False, # (Make fft, pre-emphasis, mel filter trainable if True)
    ```
3. [MFCC](apple_pytorch_speech_features/feature_extraction.py#368): MFCC Extraction = FBank --> DCT --> Roll features
    ```python
    sr=16000, # (Sample rate of input wav signal)
    winlen=400, # (Window length used for FFT/DFT/STFT)
    winstep=160, # (Window step used for FFT/DFT/STFT)
    mel_filt_path=None, # (Path of pre-calculated mel filter coefficients)
    mel_min=64, # (minimum frequency (Hz) for mel filter coefficient calculation)
    mel_max=8000, # (maximum frequency (Hz) for mel filter coefficient calculation)
    num_mels=40, # (Number of mel filterbanks)
    num_mfccs=20, # (Number of mfcc)
    premph_k=0.97, # (Pre-emphasis coefficient)
    winfunc=np.hamming, # (window function to be used for FFT)
    remove_dc=True, # (Remove mean after windowing)
    add_noise=True, # (same as dither in Kaldi)
    do_log=True, # (Produce results in log)
    scale_spectrogram=1, # (Add any constant scale factor to be used on spectral outputs (like 1/NFFT))
    scale_fbanks=1, # (Add any constant scale factor to be used on fbank outputs like use 1/ln(10) if you want outputs in log 10 scale))
    requires_grad=False, # (Make fft, pre-emphasis, mel filter, dct-filter trainable if True)
    ```
4. [SlidingCMVN](apple_pytorch_speech_features/feature_extraction.py#460): Sliding Cmvn with a given minimum cmn size and cmn size
   ```python
    cmn_window=600, # (cmn window used to calculate mean over)
    min_cmn_window=100, # (cmn window used to calculate mean over)
    ```
5. [SubSample](apple_pytorch_speech_features/feature_extraction.py#546): SubSample features
   ```python
    stride=1, # (Subsampling parameter)
    feat_dim=40, # (N in Dimension of input features BxTxN)
    ```
6. [Splicing](apple_pytorch_speech_features/feature_extraction.py#546): Splice features with left-context and right-cintext parameters
   ```python
    leftpad=3, # (Left context)
    rightpad=3, # (Right context)
    feat_dim=40, # (N in Dimension of input features BxTxN)
    ```
7. [GlobalCMVN](apple_pytorch_speech_features/feature_extraction.py#600): Apply CMVN/normalization using mean & std computed on whole dataset
   ```python
    cmvn_mean = <list of feat mean, len(list) = input_feat_dim>
    cmvn_std  = <list of feat std, len(list)  = input_feat_dim>
    ```


