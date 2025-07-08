import os
import torch
import json
import numpy as np
import scipy.stats

from catboost import Pool
from tqdm import tqdm

from utils.features import extract_mel_spectrogram, normalize_spec


def hjorth_params(signal: np.ndarray):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    activity = np.var(signal)
    mobility = np.sqrt(np.var(first_deriv) / activity) if activity > 0 else 0
    complexity = (
        np.sqrt(np.var(second_deriv) / np.var(first_deriv))
        / mobility if mobility > 0 and np.var(first_deriv) > 0 else 0
    )
    return activity, mobility, complexity


def signal_entropy(x: np.ndarray, bins: int = 64):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def extract_signal_scalar_features(samples: np.ndarray) -> dict:
    if len(samples) == 0:
        return {}

    features = {}

    features["mean"] = np.mean(samples)
    features["std"] = np.std(samples)
    features["max"] = np.max(samples)
    features["min"] = np.min(samples)
    features["skew"] = scipy.stats.skew(samples)
    features["kurtosis"] = scipy.stats.kurtosis(samples)
    features["energy"] = np.sum(samples ** 2)
    features["rms"] = np.sqrt(np.mean(samples ** 2))
    features["abs_mean"] = np.mean(np.abs(samples))
    features["max_abs"] = np.max(np.abs(samples))
    features["ptp"] = np.ptp(samples)
    features["zero_crossings"] = np.sum(np.diff(np.sign(samples)) != 0)

    if features["rms"] > 0:
        features["peak_to_rms"] = features["max_abs"] / features["rms"]
    else:
        features["peak_to_rms"] = 0

    first_derivative = np.diff(samples)
    features["mean_abs_deriv"] = np.mean(np.abs(first_derivative))
    features["std_deriv"] = np.std(first_derivative)

    hjorth_act, hjorth_mob, hjorth_comp = hjorth_params(samples)
    features["hjorth_activity"] = hjorth_act
    features["hjorth_mobility"] = hjorth_mob
    features["hjorth_complexity"] = hjorth_comp

    features["entropy"] = signal_entropy(samples)

    noise_est = np.std(samples - np.mean(samples))
    signal_power = np.mean(samples ** 2)
    features["snr"] = 10 * np.log10(signal_power / (noise_est ** 2 + 1e-10))

    return features


def extract_spectral_scalar_features(samples, sample_rate, n_mels=64, n_fft=512, hop_length=64,
                                     band_count=4) -> dict:
    log_mel_spec = extract_mel_spectrogram(torch.tensor(samples).unsqueeze(0), sample_rate, n_mels, n_fft, hop_length)

    features = {}
    mel_spec = log_mel_spec.squeeze().detach().cpu().numpy()
    band_size = n_mels // band_count

    for i in range(band_count):
        start = i * band_size
        end = (i + 1) * band_size if i < band_count - 1 else n_mels
        band = mel_spec[start:end, :]
        band_flat = band.flatten()
        prefix = f"band{i}_"
        features[prefix + "mean"] = np.mean(band_flat)
        features[prefix + "std"] = np.std(band_flat)

        features[prefix + "max"] = np.max(band_flat)
        features[prefix + "min"] = np.min(band_flat)
        features[prefix + "energy"] = np.sum(band_flat)
        features[prefix + "skew"] = scipy.stats.skew(band_flat)
        features[prefix + "kurtosis"] = scipy.stats.kurtosis(band_flat)
        features[prefix + "entropy"] = -np.sum((p := band_flat / (np.sum(band_flat) + 1e-9)) * np.log2(p + 1e-9))
        features[prefix + "contrast"] = np.max(band_flat) - np.min(band_flat)

        flux = np.sqrt(np.sum(np.diff(band, axis=1) ** 2, axis=0))
        features[prefix + "spectral_flux_mean"] = np.mean(flux)
        features[prefix + "spectral_flux_std"] = np.std(flux)

        max_freq_idx = np.argmax(np.mean(band, axis=1))
        features[prefix + "max_freq_idx_norm"] = max_freq_idx / band.shape[0]

        time_energy = np.sum(band, axis=0)
        max_time_idx = np.argmax(time_energy)
        features[prefix + "max_time_idx_norm"] = max_time_idx / band.shape[1]

        if band.shape[1] > 1:
            x = np.arange(band.shape[1])
            y = time_energy
            A = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            features[prefix + "temporal_slope"] = slope

        features[prefix + "temporal_energy_mean"] = np.mean(time_energy)
        features[prefix + "temporal_energy_std"] = np.std(time_energy)
        features[prefix + "delta_energy"] = time_energy[-1] - time_energy[0]

        features[prefix + "nonzero_fraction"] = np.count_nonzero(band) / len(band_flat)

    return features


def process_dataset_split(root_path: str, split: str, sample_rate: int = 16000,
                          segment_duration: float = 0.2, overlap: float = 0.15):
    split_path = os.path.join(root_path, split)
    print(f"Processing split: {split_path}")
    class_folders = sorted(os.listdir(split_path))
    print(f"Found class folders: {class_folders}")

    all_features = []
    all_labels = []
    feature_names = None
    class_mapping = {}

    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        class_mapping[class_idx] = class_name

        file_list = [f for f in os.listdir(class_path) if f.endswith(".wav")]
        print(f"\nProcessing class '{class_name}' with {len(file_list)} files")

        for file_name in tqdm(file_list, desc=f"[{split}] Class: {class_name}", unit="file"):
            file_path = os.path.join(class_path, file_name)
            try:
                samples, sr = load_audio(file_path, sample_rate=sample_rate)
                if sr != sample_rate:
                    continue
            except Exception:
                continue

            window_size = int(segment_duration * sample_rate)
            step_size = int((segment_duration - overlap) * sample_rate)
            total_samples = len(samples)

            for start in range(0, total_samples - window_size + 1, step_size):
                segment = samples[start:start + window_size]
                signal_feats = extract_signal_scalar_features(segment)
                spectral_feats = extract_spectral_scalar_features(segment, sample_rate)

                combined = {**signal_feats, **spectral_feats}

                if feature_names is None:
                    feature_names = sorted(combined.keys())

                values = [combined[k] for k in feature_names]
                all_features.append(np.array(values, dtype=np.float32))
                all_labels.append(class_idx)

    return (
        np.stack(all_features),
        np.array(all_labels),
        feature_names,
        class_mapping
    )


def save_dataset_splits(root_path: str, output_path: str, sample_rate: int = 16000,
                        segment_duration: float = 0.2, overlap: float = 0.15):
    os.makedirs(output_path, exist_ok=True)

    for split in ["train", "valid", "test"]:
        print(f"\nProcessing split: {split}")
        features, labels, feature_names, class_mapping = process_dataset_split(
            root_path=root_path,
            split=split,
            sample_rate=sample_rate,
            segment_duration=segment_duration,
            overlap=overlap
        )

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        feature_names = np.array(feature_names)

        output_file = os.path.join(output_path, f"{split}.npz")

        class_mapping_str = json.dumps(class_mapping)

        np.savez_compressed(
            output_file,
            features=features,
            labels=labels,
            feature_names=feature_names,
            class_mapping=class_mapping_str
        )

        print(f"Saved {split}: {features.shape[0]} samples -> {output_file}")


def load_catboost_data(npz_path: str, return_pool: bool = True):
    data = np.load(npz_path)
    X = data["features"]
    y = data["labels"]

    if return_pool:
        return Pool(data=X, label=y)
    else:
        return X, y


if __name__ == "__main__":
    import time
    from features import load_audio

    sample_rate = 16000
    time_step = .2
    root = r'D:\Projects\Python\drone-detection-c\dataset\clean-baseline-audios'
    scalar_features = r'D:\Projects\Python\drone-detection-c\dataset\scalar_features'
    save_dataset_splits(root, 'D:\Projects\Python\drone-detection-c\dataset\scalar_features')
    # X, y = load_catboost_data(os.path.join(scalar_features, 'train.npz'), return_pool=False)
    # print(len(X[0]))
    # print(X[:10])
    # print(y[:10])

    # samples, sr = load_audio(
    #     r"D:\Projects\Python\drone-detection-c\dataset\clean-baseline-audios\train\free_space\2025-02-26 14-43-04.wav",
    #     sample_rate=sample_rate, return_tensor=False)
    # samples = np.array(samples[60 * int(time_step * sample_rate):61 * int(time_step * sample_rate)])
    #
    # start = time.time()
    # print(len(extract_signal_scalar_features(samples)))
    # end_old = time.time()
    # print('Signal ', end_old - start)
    #
    # print(len(extract_spectral_scalar_features(samples, sample_rate)))
    # end = time.time()
    #
    # print('Spectral', end - start)
    #
    # print(end + end_old - 2 * start, 0.05 - (end + end_old - 2 * start))
