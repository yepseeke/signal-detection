import os
import torch
import numpy as np

from tqdm import tqdm

from utils.features import (load_audio, extract_mel_spectrogram, extract_hpss_components, extract_mfcc,
                                extract_spectral_contrast, extract_zero_crossing_rate, extract_spectral_centroid,
                                extract_spectral_bandwidth, extract_rms_energy)


def extract_features_from_segment(
        file_path: str,
        segment_duration: float = 0.2,
        sample_rate: int = 16000,
        hop_duration: float = 0.1,
        n_mfcc=13,
        n_mels=64,
        n_fft=512,
        hop_length=64,
        n_bands=6,
        return_type: str = 'numpy'
):
    assert return_type in ['numpy', 'tensor'], "return_type должен быть 'numpy' или 'tensor'"

    samples, sr = load_audio(file_path, sample_rate=sample_rate, mono=True, return_tensor=False)
    total_samples = samples.shape[0]

    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)

    mel_list = []
    mfcc_list = []
    contrast_list = []

    for start in range(0, total_samples - segment_samples + 1, segment_samples):
        end = start + segment_samples
        segment = samples[start:end]

        segment_tensor = torch.tensor(segment).unsqueeze(0)

        mel = extract_mel_spectrogram(segment_tensor, sr, n_mels, n_fft, hop_length).numpy()
        mfcc = extract_mfcc(segment_tensor, sr, n_mfcc, n_mels, n_fft, hop_length).numpy()
        contrast = extract_spectral_contrast(segment, sr, n_fft, hop_length, n_bands, return_type='tensor')

        mel_list.append(mel)
        mfcc_list.append(mfcc)
        contrast_list.append(contrast)

    mel_array = np.stack(mel_list)
    mfcc_array = np.stack(mfcc_list)
    contrast_array = np.stack(contrast_list)

    if return_type == 'tensor':
        mel_array = torch.tensor(mel_array)
        mfcc_array = torch.tensor(mfcc_array)
        contrast_array = torch.tensor(contrast_array)

    return mel_array, mfcc_array, contrast_array


def safe_save_npz(output_folder, mel_array, mfcc_array, contrast_array):
    save_path = os.path.join(output_folder, "features.npz")

    for name, arr in [('mel', mel_array), ('mfcc', mfcc_array), ('contrast', contrast_array)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"{name} contains NaN or Inf")

    tmp_path = save_path + ".tmp.npz"
    np.savez_compressed(tmp_path, mel=mel_array, mfcc=mfcc_array, contrast=contrast_array)

    try:
        with np.load(tmp_path) as data:
            _ = data['mel']
    except Exception as e:
        os.remove(tmp_path)
        raise IOError(f"File {tmp_path} is corrupted after writing: {e}")

    os.replace(tmp_path, save_path)


def process_audio_folder(
        input_folder: str,
        output_folder: str,
        segment_duration: float = 0.2,
        sample_rate: int = 16000,
        hop_duration: float = 0.1,
        n_mfcc=13,
        n_mels=64,
        n_fft=512,
        hop_length=64,
        n_bands=6,
        return_type: str = 'numpy'
):
    os.makedirs(output_folder, exist_ok=True)

    all_mel = []
    all_mfcc = []
    all_contrast = []

    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]

    for file in tqdm(audio_files, desc="Processing audio files"):
        file_path = os.path.join(input_folder, file)

        try:
            mel, mfcc, contrast = extract_features_from_segment(
                file_path=file_path,
                segment_duration=segment_duration,
                sample_rate=sample_rate,
                hop_duration=hop_duration,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                n_bands=n_bands,
                return_type=return_type
            )

            all_mel.append(mel)
            all_mfcc.append(mfcc)
            all_contrast.append(contrast)

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    mel_array = np.concatenate(all_mel, axis=0)
    mfcc_array = np.concatenate(all_mfcc, axis=0)
    contrast_array = np.concatenate(all_contrast, axis=0)

    safe_save_npz(output_folder, mel_array, mfcc_array, contrast_array)

    print(f"Saved: {output_folder}")


def process_audios_into_arrays(input_path: str,
                               output_path: str,
                               segment_duration: float = 0.2,
                               sample_rate: int = 16000,
                               hop_duration: float = 0.1,
                               n_mfcc=13,
                               n_mels=64,
                               n_fft=512,
                               hop_length=64,
                               n_bands=6,
                               return_type: str = 'numpy'):
    folders = ['train', 'valid', 'test']

    for folder in folders:
        input_folder = os.path.join(input_path, folder)
        output_folder = os.path.join(output_path, folder)

        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Folder '{folder}' not found in '{input_path}'")

        os.makedirs(output_folder, exist_ok=True)

        classes = os.listdir(input_folder)
        for cls in classes:
            class_input_folder = os.path.join(input_folder, cls)
            class_output_folder = os.path.join(output_folder, cls)

            if not os.path.isdir(class_input_folder):
                continue

            print(f"Processing class: {cls}")
            process_audio_folder(class_input_folder, class_output_folder, segment_duration, sample_rate,
                                 hop_duration, n_mfcc, n_mels, n_fft, hop_length, n_bands, return_type)


def load_matrix_features_npz(npz_path):
    data = np.load(npz_path)

    mel = data['mel']
    mfcc = data['mfcc']
    contrast = data['contrast']

    return mel, mfcc, contrast


def load_dataset_from_npz(root_dir):
    datasets = {'train': [], 'valid': [], 'test': []}
    class_to_idx = {}

    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Folder {split} not found in {root_dir}")

        for class_name in sorted(os.listdir(split_path)):
            class_dir = os.path.join(split_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)

            npz_path = os.path.join(class_dir, 'features.npz')
            if not os.path.exists(npz_path):
                print(f"[Warning] No features.npz file in {class_dir}, skipped.")
                continue

            mel, mfcc, contrast = load_matrix_features_npz(npz_path)

            for i in range(len(mel)):
                datasets[split].append((
                    mel[i], mfcc[i], contrast[i],
                    class_to_idx[class_name]
                ))

    idx_to_class = {class_to_idx[class_name]: class_name for class_name in class_to_idx.keys()}
    return datasets['train'], datasets['valid'], datasets['test'], class_to_idx, idx_to_class


if __name__ == '__main__':
    process_audios_into_arrays(r'/home/pavel/Projects/github/signal-detection/dataset/baseline/',
                               r'/home/pavel/Projects/github/signal-detection/dataset/baseline-arrays', hop_duration=0)
    # from src.utils.plot import plot_feature_matrix

    # matrix = \
    # load_features_npz(r'D:\Projects\Python\drone-detection-c\dataset\baseline-arrays\train\small_copter\features.npz')[
    #     0][2]
    # plot_feature_matrix(matrix)
