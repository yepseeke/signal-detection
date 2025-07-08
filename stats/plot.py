import librosa
import time

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch import Tensor


def plot_waveform(samples, sample_rate, title="Waveform"):
    plt.figure(figsize=(10, 3))
    time = np.arange(samples.shape[-1]) / sample_rate
    plt.plot(time, samples)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()


def plot_spectrogram(spec, title="Spectrogram", ylabel="Frequency"):
    plt.figure(figsize=(10, 4))
    if isinstance(spec, Tensor):
        spec = spec.squeeze().cpu().numpy()
    librosa.display.specshow(spec, sr=129000, x_axis='time', y_axis='mel')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()


def plot_mfcc(mfcc, title="MFCC"):
    plt.figure(figsize=(10, 4))
    if isinstance(mfcc, Tensor):
        mfcc = mfcc.squeeze().cpu().numpy()
    librosa.display.specshow(mfcc, x_axis='time')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()


def plot_chroma(chroma, title="Chroma"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()


def plot_scalar_feature(feature, title="Feature", ylabel="Value"):
    plt.figure(figsize=(10, 3))
    if isinstance(feature, Tensor):
        feature = feature.squeeze().cpu().numpy()
    plt.plot(feature.T)
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()


def plot_spectral_contrast(contrast, sample_rate, hop_length=64, title='Spectral Contrast'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(contrast, x_axis='time', sr=sample_rate, hop_length=hop_length)
    plt.colorbar(label='Contrast (dB)')
    plt.title(title)
    plt.tight_layout()


def print_temporal_features(features_dict):
    print("Temporal features:")
    for key, value in features_dict.items():
        print(f"  {key:10s}: {value:.4f}")


def plot_hpss(harmonic, percussive, sample_rate):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    librosa.display.waveshow(harmonic, sr=sample_rate)
    plt.title("Harmonic Component")

    plt.subplot(1, 2, 2)
    librosa.display.waveshow(percussive, sr=sample_rate)
    plt.title("Percussive Component")

    plt.tight_layout()


def print_temporal_features_to_plot(features_dict, title="Temporal Features"):
    plt.figure(figsize=(5, 3))
    plt.axis('off')
    text = "\n".join(f"{k:10s}: {v:.4f}" for k, v in features_dict.items())
    plt.text(0.1, 0.5, text, fontsize=12)
    plt.title(title)
    plt.tight_layout()


def visualize_all_features_to_pdf(samples, sr, save_path="features_summary.pdf"):
    harmonic, percussive = librosa.effects.hpss(samples)
    modes = {
        'Original': samples,
        'Harmonic': harmonic,
        'Percussive': percussive,
    }

    with PdfPages(save_path) as pdf:
        for name, signal in modes.items():
            print(f"\nProcessing: {name}")
            signal_tensor = torch.tensor(signal).unsqueeze(0)

            start = time.time()
            mel = extract_mel_spectrogram(signal_tensor, sr)
            print(f"[{name}] Mel spectrogram: {time.time() - start:.3f} sec")
            plot_spectrogram(mel, title=f"{name} - Mel Spectrogram")
            pdf.savefig()
            plt.close()

            # start = time.time()
            # mfcc = extract_mfcc(signal_tensor, sr)
            # print(f"[{name}] MFCC: {time.time() - start:.3f} sec")
            # plot_mfcc(mfcc, title=f"{name} - MFCC")
            # pdf.savefig(); plt.close()

            start = time.time()
            zcr = extract_zero_crossing_rate(signal)
            print(f"[{name}] ZCR: {time.time() - start:.3f} sec")
            plot_scalar_feature(zcr, title=f"{name} - Zero-Crossing Rate", ylabel="ZCR")
            pdf.savefig()
            plt.close()

            start = time.time()
            centroid = extract_spectral_centroid(signal, sr) / (sr // 2)
            print(f"[{name}] Spectral Centroid: {time.time() - start:.3f} sec")
            plot_scalar_feature(centroid, title=f"{name} - Spectral Centroid", ylabel="Hz")
            pdf.savefig()
            plt.close()

            start = time.time()
            bandwidth = extract_spectral_bandwidth(signal, sr) / (sr // 2)
            print(f"[{name}] Spectral Bandwidth: {time.time() - start:.3f} sec")
            plot_scalar_feature(bandwidth, title=f"{name} - Spectral Bandwidth", ylabel="Hz")
            pdf.savefig()
            plt.close()

            start = time.time()
            rms = extract_rms_energy(signal)
            print(f"[{name}] RMS Energy: {time.time() - start:.3f} sec")
            plot_scalar_feature(rms, title=f"{name} - RMS Energy", ylabel="Amplitude")
            pdf.savefig()
            plt.close()

            # start = time.time()
            # contrast = extract_spectral_contrast(signal, sr, return_type='numpy')
            # print(f"[{name}] Spectral Contrast: {time.time() - start:.3f} sec")
            # plot_spectral_contrast(contrast, sr, title=f"{name} - Spectral Contrast")
            # pdf.savefig(); plt.close()

            start = time.time()
            temporal_feats = extract_temporal_features(signal)
            print(f"[{name}] Temporal features: {time.time() - start:.3f} sec")
            print_temporal_features_to_plot(temporal_feats, title=f"{name} - Temporal Stats")
            pdf.savefig()
            plt.close()


def plot_feature_matrix(matrix, title="Feature Matrix", xlabel="Time Frames", ylabel="Frequency Bins", cmap="viridis"):
    if hasattr(matrix, 'numpy'):
        matrix = matrix.numpy()

    if matrix.ndim == 3 and matrix.shape[0] == 1:  # [1, F, T] -> [F, T]
        matrix = matrix[0]

    plt.figure(figsize=(10, 4))
    plt.imshow(matrix, aspect='auto', origin='lower', cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch

    from utils.features import (load_audio, extract_mel_spectrogram, extract_mfcc, extract_chroma,
                                    extract_zero_crossing_rate, extract_spectral_centroid, extract_spectral_bandwidth,
                                    extract_rms_energy,
                                    extract_spectral_contrast, extract_temporal_features, extract_hpss_components,
                                extract_sequence_features_per_mode, scale_features_per_mode)

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    sample_rate = 16000
    time_step = 0.2

    samples, sr = load_audio(r"D:\Projects\Python\drone-detection-c\dataset\clean-baseline-audios\train\small_copter\2024-05-22 13-55-55_converted4.wav",
                             sample_rate=sample_rate, return_tensor=False)
    samples = samples[60 * int(time_step * sample_rate):61 * int(time_step * sample_rate)]
    features = extract_sequence_features_per_mode(
        samples,
        sample_rate,
        frame_length=512,
        hop_length=64,
        n_fft=1024
    )
    plot_waveform(samples, sr)
    plt.show()
    print(features['percussive'].shape)
    scaled = scale_features_per_mode(features, method='minmax')

    plt.plot(features['original'])
    plt.show()
    plt.plot(scaled['original'])
    plt.show()
    visualize_all_features_to_pdf(samples, sr)

    mel = extract_mel_spectrogram(torch.tensor(samples).unsqueeze(0), sr)
    plot_spectrogram(mel, title="Mel Spectrogram")
    plt.show()
    #
    # mfcc = extract_mfcc(torch.tensor(samples).unsqueeze(0), sr)
    # print(mfcc.shape)
    # plot_mfcc(mfcc)
    # plt.show()

    zcr = extract_zero_crossing_rate(samples, frame_length=512)
    plot_scalar_feature(zcr, title="Zero-Crossing Rate", ylabel="ZCR")
    print(zcr.shape)
    plt.show()

    centroid = extract_spectral_centroid(samples, sr, n_fft=1024)
    plot_scalar_feature(centroid, title="Spectral Centroid", ylabel="Hz")
    print(centroid.shape)
    plt.show()

    bandwidth = extract_spectral_bandwidth(samples, sr)
    plot_scalar_feature(bandwidth, title="Spectral Bandwidth", ylabel="Hz")
    print(bandwidth.shape)
    plt.show()

    rms = extract_rms_energy(samples)
    plot_scalar_feature(rms, title="RMS Energy", ylabel="Amplitude")
    print(rms.shape)
    plt.show()

    contrast = extract_spectral_contrast(samples, sr, return_type='numpy', expand_dims=False)
    print(contrast.shape)
    plot_spectral_contrast(contrast, sr)
    plt.show()

    temporal_feats = extract_temporal_features(samples)
    print_temporal_features(temporal_feats)
