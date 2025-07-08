import os
import time
import torch
import io

import PIL.Image
import seaborn as sns

from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from nn.model import get_model


def evaluate_metrics(y_true, y_pred, class_names=None, verbose=True):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print("Per-class metrics:")
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            cls_name = class_names[i] if class_names else f"Class {i}"
            print(f"{cls_name}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

        print(f"\nOverall Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        print("\nConfusion Matrix:")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": {
            class_names[i] if class_names else str(i): {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i]
            }
            for i in range(len(f1))
        },
        "confusion_matrix": cm
    }


def measure_inference_time(model, dataloader, device, use_amp=True, max_batches=10):
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            mel_input = batch['mel'].to(device)

            start = time.time()
            with autocast(enabled=use_amp, device_type=device):
                _ = model(mel_input)
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time


def validate_model(model, valid_loader, criterion, device, epoch=None, num_epochs=None, use_amp=True):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    desc = f"Epoch {epoch + 1}/{num_epochs} [Valid]" if epoch is not None and num_epochs is not None else "Validation"

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=desc):
            mel_input = batch['mel'].to(device)
            targets = batch['label'].to(device)

            with autocast(enabled=use_amp, device_type=device):
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

    return avg_val_loss, val_acc, all_targets, all_preds


def train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        num_epochs=10,
        save_path=None,
        idx_to_class=None,
        scheduler=None,
        clip_grad_norm=1.0,
        use_amp=True,
        model_name=None
):
    model.to(device)
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    if model_name is None:
        model_name_loss = "best_loss"
        model_name_acc = "best_acc"
        model_name = "unnamed_model"
    else:
        model_name_loss = f"{model_name}_best_loss"
        model_name_acc = f"{model_name}_best_acc"

    writer = SummaryWriter(log_dir=f"runs/{model_name}")

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            mel_input, targets = batch['mel'].to(device), batch['label'].to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp, device_type=device):
                outputs = model(mel_input)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            writer.add_scalar("Train/GradientNorm", total_norm**0.5, epoch)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        avg_val_loss, val_acc, all_targets, all_preds = validate_model(
            model, valid_loader, criterion, device, epoch, num_epochs, use_amp
        )
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}: "
              f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}, "
              f"Time = {epoch_time:.1f}s\n")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LR/group_{i}", param_group['lr'], epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        if idx_to_class:
            class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        else:
            class_names = [str(i) for i in range(len(set(all_targets)))]

        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = PIL.Image.open(buf)
        image = np.array(image)
        writer.add_image("Val/Confusion_Matrix", image, epoch, dataformats='HWC')
        plt.close(fig)

        evaluate_metrics(all_targets, all_preds, idx_to_class)

        if scheduler:
            scheduler.step()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_loss_path = os.path.join(save_path, f"{model_name_loss}.pth")
                torch.save(model.state_dict(), best_loss_path)
                print(f"✅ Saved best model by val loss: {best_val_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc_path = os.path.join(save_path, f"{model_name_acc}.pth")
                torch.save(model.state_dict(), best_acc_path)
                print(f"✅ Saved best model by val accuracy: {best_val_acc:.4f}")

    writer.close()


if __name__ == '__main__':
    import numpy as np

    from model import MultiBranchNet
    from dataset import create_dataloaders
    from utils.preprocess import load_dataset_from_npz

    from collections import Counter
    from torch.optim import AdamW
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from sklearn.utils.class_weight import compute_class_weight

    from matplotlib import pyplot as plt

    device = 'cuda:0'

    train_data, valid_data, test_data, class_to_idx, idx_to_class = load_dataset_from_npz(
        r'D:\Projects\Python\drone-detection-c\dataset\clean-baseline-arrays')

    class_labels = [item[-1] for item in train_data]
    class_counts = Counter(class_labels)
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    # class_weights = [total_samples / (class_counts[i] * num_classes) for i in range(num_classes)]
    # class_weights = torch.tensor(class_weights, dtype=torch.float32)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=class_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(class_counts)

    train_loader, valid_loader, test_loader = create_dataloaders(train_data, valid_data, test_data, batch_size=128)

    model = get_model('resnet18', 4)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device=device,
        num_epochs=25,
        save_path=r'D:\Projects\Python\drone-detection-c\checkpoints',
        idx_to_class=idx_to_class
    )

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


    def collect_predictions(model, val_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data['mel'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels


    preds, labels = collect_predictions(model, valid_loader, device)

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
