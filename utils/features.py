import torch
import scipy
import torchaudio
import librosa

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def rms_normalize(samples, target_rms=0.1):
    rms = (samples ** 2).mean().sqrt()
    return samples * (target_rms / (rms + 1e-6))


def load_audio(
        file_path: str,
        sample_rate: int = None,
        mono: bool = True,
        return_tensor: bool = False,
        target_rms=None
):
    waveform, original_sample_rate = torchaudio.load(file_path, normalize=True)

    if sample_rate is None:
        sample_rate = original_sample_rate

    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, sample_rate)

    if mono and waveform.dim() > 1:
        waveform = waveform.mean(dim=0)

    if target_rms:
        waveform = rms_normalize(waveform, target_rms)

    if return_tensor:
        return waveform, sample_rate
    else:
        return waveform.numpy(), sample_rate


def normalize_spec(spec, min_value=-80.0, max_value=0.0):
    return (spec - min_value) / (max_value - min_value)


# librosa way to get mel spectrogram
def extract_mel_spectrogram_librosa(samples, sample_rate, n_mels=64, n_fft=512, hop_length=64):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().numpy()

    mel_spec = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec


def power_to_db_torch(mel_spec, top_db=80.0):
    amin = 1e-10
    power = mel_spec.clamp(min=amin)

    ref = power.amax(dim=(-1, -2), keepdim=True)
    log_spec = 10.0 * (power / ref).log10()

    if top_db is not None:
        log_spec = torch.clamp(log_spec, min=log_spec.amax(dim=(-1, -2), keepdim=True) - top_db)

    return log_spec


def extract_mel_spectrogram(samples, sample_rate, n_mels=64, n_fft=512, hop_length=64, normalize=True):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_spec = transform(samples)
    log_mel_spec = power_to_db_torch(mel_spec, top_db=80.0)

    if normalize:
        log_mel_spec = normalize_spec(log_mel_spec, min_value=-80, max_value=0)

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


def extract_sequence_features_per_mode(
        samples,
        sample_rate=16000,
        frame_length=512,
        hop_length=64,
        n_fft=512
):
    harmonic, percussive = librosa.effects.hpss(samples)
    modes = {
        'original': samples,
        'harmonic': harmonic,
        'percussive': percussive,
    }

    all_features = {}

    for name, signal in modes.items():
        zcr = extract_zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length)
        centroid = extract_spectral_centroid(signal, sample_rate, n_fft=n_fft, hop_length=hop_length) / sample_rate // 2
        bandwidth = extract_spectral_bandwidth(signal, sample_rate, n_fft=n_fft,
                                               hop_length=hop_length) / sample_rate // 2
        rms = extract_rms_energy(signal, frame_length=frame_length, hop_length=hop_length)

        print(rms)
        feature_matrix = np.vstack([
            zcr,
            centroid,
            bandwidth,
            rms
        ]).T

        all_features[name] = feature_matrix

    return all_features


def scale_features_per_mode(
        feature_dict: dict,
        method: str = 'standard'
) -> dict:
    assert method in ['standard', 'minmax'], "standard' or 'minmax'"

    scaler_class = StandardScaler if method == 'standard' else MinMaxScaler
    scaled_dict = {}

    for mode_name, features in feature_dict.items():
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        scaler = scaler_class()
        scaled_features = scaler.fit_transform(features)
        scaled_dict[mode_name] = scaled_features

    return scaled_dict
