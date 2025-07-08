import torch

from torch.utils.data import Dataset, DataLoader


class AudioFeatureDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel, mfcc, contrast, label = self.data[idx]

        mel = torch.tensor(mel, dtype=torch.float32)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        contrast = torch.tensor(contrast, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            mel, mfcc, contrast = self.transform(mel, mfcc, contrast)

        return {
            'mel': mel,
            'mfcc': mfcc,
            'contrast': contrast,
            'label': label
        }


def create_dataloaders(train_data, valid_data, test_data, batch_size=32, shuffle=True, transform=None):
    # if transform is None:
    #     transform = ResizeFeatures(size=(64, 51))

    train_ds = AudioFeatureDataset(train_data, transform)
    valid_ds = AudioFeatureDataset(valid_data, transform)
    test_ds = AudioFeatureDataset(test_data, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
