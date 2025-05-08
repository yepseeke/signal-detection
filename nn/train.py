import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support, classification_report

from src.nn.model import get_model


def evaluate_metrics(y_true, y_pred, class_names=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    print("Per-class metrics:")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        cls_name = class_names[i] if class_names else f"Class {i}"
        print(f"{cls_name}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")


def train_cnn_model(
        model,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion,
        optimizer,
        device,
        num_epochs=10,
        save_path=None
):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for train_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            mel_input, mfcc_input, contrast_input, targets = (train_data['mel'], train_data['mfcc'],
                                                              train_data['contrast'], train_data['label'])
            mel_input, mfcc_input, contrast_input = (mel_input.to(device), mfcc_input.to(device),
                                                     contrast_input.to(device))
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(mel_input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for valid_data in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]"):
                mel_input, mfcc_input, contrast_input, targets = (valid_data['mel'], valid_data['mfcc'],
                                                                  valid_data['contrast'], valid_data['label'])
                mel_input, mfcc_input, contrast_input = (mel_input.to(device), mfcc_input.to(device),
                                                         contrast_input.to(device))
                targets = targets.to(device)

                outputs = model(mel_input, mfcc_input, contrast_input)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

                all_targets.extend(targets.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

        avg_val_loss = val_loss / total
        val_acc = correct / total

        print(f"\nEpoch {epoch + 1}: "
              f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}\n")

        evaluate_metrics(all_targets, all_preds)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")


def train_model(
        model,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion,
        optimizer,
        device,
        num_epochs=10,
        save_path=None,
        idx_to_class=None
):
    """
    Обучает и валидирует модель, используя только mel_input из батчей данных.

    Args:
        model: Модель PyTorch (должна принимать только один тензор на входе).
        train_loader: DataLoader для тренировочных данных.
        valid_loader: DataLoader для валидационных данных.
        criterion: Функция потерь.
        optimizer: Оптимизатор.
        device: Устройство для обучения ('cuda' или 'cpu').
        num_epochs: Количество эпох обучения.
        save_path: Путь для сохранения весов модели.
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for train_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            mel_input, targets = train_data['mel'], train_data['label']

            mel_input = mel_input.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(mel_input)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for valid_data in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]"):
                mel_input, targets = valid_data['mel'], valid_data['label']

                mel_input = mel_input.to(device)
                targets = targets.to(device)

                outputs = model(mel_input)

                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

                all_targets.extend(targets.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

        avg_val_loss = val_loss / total
        val_acc = correct / total

        print(f"\nEpoch {epoch + 1}: "
              f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}\n")

        evaluate_metrics(all_targets, all_preds, idx_to_class)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    from model import MultiBranchNet
    from dataset import create_dataloaders
    from src.utils.preprocess import load_dataset_from_npz

    from matplotlib import pyplot as plt

    # def show_feature_map(batch_tensor, index=0, title="Feature Map"):
    #     """
    #     Показывает один из тензоров из батча.
    #
    #     Parameters:
    #     - batch_tensor: torch.Tensor формы (B, 1, H, W)
    #     - index: индекс элемента в батче, который нужно нарисовать
    #     - title: заголовок рисунка
    #     """
    #     assert batch_tensor.dim() == 4 and batch_tensor.shape[1] == 1, \
    #         "Ожидается тензор формы (B, 1, H, W)"
    #
    #     feature_map = batch_tensor[index, 0].cpu().numpy()
    #
    #     plt.figure(figsize=(10, 4))
    #     plt.imshow(feature_map, aspect='auto', origin='lower', cmap='viridis')
    #     plt.colorbar()
    #     plt.title(title)
    #     plt.xlabel("Time")
    #     plt.ylabel("Frequency")
    #     plt.tight_layout()
    #     plt.show()

    device = 'cpu'

    train_data, valid_data, test_data, class_to_idx, idx_to_class = load_dataset_from_npz(
        r"D:\Projects\Python\drone-detection-c\dataset\baseline-arrays-no-noise")

    print(test_data[0])
    class_labels = [item[-1] for item in train_data]

    from collections import Counter
    class_counts = Counter(class_labels)

    print(class_counts)

    train_loader, valid_loader, test_loader = create_dataloaders(train_data, valid_data, test_data, batch_size=16)
    batch = next(iter(train_loader))
    print(batch['mel'].shape, batch['mfcc'].shape, batch['contrast'].shape, batch['label'])

    # show_feature_map(batch['mel'], index=0, title="Mel Spectrogram After Interpolation")

    model = get_model('mobilenetv2', 3)
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device=device,
        num_epochs=20,
        save_path=r'D:\Projects\Python\drone-detection-c\checkpoints',
        idx_to_class=idx_to_class
    )
