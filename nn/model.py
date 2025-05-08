import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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

        dummy_input_vgg = torch.randn(1, 1, 64, 51)
        output_features = model.features(dummy_input_vgg)

        in_features_vgg = output_features.view(output_features.size(0), -1).shape[1]

        model.classifier = nn.Sequential(
            nn.Linear(in_features_vgg, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        return model

    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
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
