import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from nn.train import evaluate_metrics


class SimpleCNN(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 7, 51]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 32, 3, 25]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 3, 25]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


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
