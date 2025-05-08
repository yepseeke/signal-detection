import torch
import scipy
import torchaudio
import librosa

import numpy as np


def load_audio(
        file_path: str,
        sample_rate: int = None,
        mono: bool = True,
        return_tensor: bool = False
):
    waveform, original_sample_rate = torchaudio.load(file_path, normalize=True)

    if sample_rate is None:
        sample_rate = original_sample_rate

    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, sample_rate)

    if mono and waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    if return_tensor:
        return waveform, sample_rate
    else:
        return waveform.numpy(), sample_rate


def extract_mel_spectrogram(samples, sample_rate, n_mels=64, n_fft=512, hop_length=64):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    mel_spec = transform(samples)
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    return log_mel_spec


def extract_mfcc(samples, sample_rate, n_mfcc=30, n_mels=64, n_fft=512, hop_length=64):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels
        }
    )

    return transform(samples)


def extract_chroma(samples, sample_rate, n_fft=512, hop_length=64):
    stft = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop_length))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    return chroma


def extract_zero_crossing_rate(samples, frame_length=512, hop_length=64):
    zcr = librosa.feature.zero_crossing_rate(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return zcr


def extract_spectral_centroid(samples, sample_rate, n_fft=512, hop_length=64):
    centroid = librosa.feature.spectral_centroid(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return centroid


def extract_spectral_bandwidth(samples, sample_rate, n_fft=512, hop_length=64):
    bandwidth = librosa.feature.spectral_bandwidth(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return bandwidth


def extract_rms_energy(samples, frame_length=512, hop_length=64):
    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return rms


def extract_spectral_contrast(
        samples,
        sample_rate,
        n_fft=512,
        hop_length=64,
        n_bands=6,
        return_type: str = 'tensor',
        expand_dims: bool = True

):
    assert return_type in ['tensor', 'numpy'], "return_type  'tensor' or 'numpy'"


    contrast = librosa.feature.spectral_contrast(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands
    )

    if expand_dims:
        contrast = contrast[np.newaxis, ...]

    if return_type == 'tensor':
        contrast = torch.tensor(contrast, dtype=torch.float32)

    return contrast


def extract_temporal_features(samples):
    return {
        'mean': np.mean(samples),
        'std': np.std(samples),
        'max': np.max(samples),
        'min': np.min(samples),
        'skew': scipy.stats.skew(samples),
        'kurtosis': scipy.stats.kurtosis(samples),
        'energy': np.sum(samples ** 2)
    }


def extract_hpss_components(samples):
    harmonic, percussive = librosa.effects.hpss(samples)
    return harmonic, percussive
