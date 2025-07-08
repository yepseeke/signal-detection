import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from cnn import SimpleCNN


class MultiBranchNet(nn.Module):
    def __init__(self, out_dim=128, num_classes=4):
        super().__init__()

        self.mel_net = torchvision.models.densenet121(weights=None)
        self.mel_net.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.mfcc_net = torchvision.models.densenet121(weights=None)
        self.mfcc_net.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.contrast_net = SimpleCNN(out_dim=1024)

        self.fc = nn.Sequential(
            nn.Linear(3 * 1024, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, mel, mfcc, contrast):
        mel_feat = self.mel_net.features(mel)
        mel_feat = F.adaptive_avg_pool2d(mel_feat, (1, 1)).flatten(1)

        mfcc_feat = self.mfcc_net.features(mfcc)
        mfcc_feat = F.adaptive_avg_pool2d(mfcc_feat, (1, 1)).flatten(1)

        contrast_feat = self.contrast_net(contrast)

        x = torch.cat([mel_feat, mfcc_feat, contrast_feat], dim=1)
        return self.fc(x)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path, device: str | torch.device, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def get_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == 'baseline':
        return SimpleCNN(num_classes)

    elif model_name == 'vgg11':
        model = models.vgg11(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=num_classes)

        return model

    elif model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=None)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif model_name == 'efficientnetb0':
        model = models.efficientnet_b0(weights=None)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == '__main__':
    for model_name in ["baseline", 'resnet18', 'resnet34', 'resnet50',
                       'vgg11', 'densenet121', 'mobilenetv2', 'efficientnetb0']:
        model = get_model(model_name, 4)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Модель: {model_name}")
        print(f"Общее количество параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,}")
        print(f"Размер модели: {total_params * 4 / (1024 ** 2):.2f} МБ (float32)")