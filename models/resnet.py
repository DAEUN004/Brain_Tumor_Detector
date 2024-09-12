
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()

        # Load pretrained ResNet50 model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = self._modify_resnet50(self.model)

    def _modify_resnet50(self, model):
        for param in model.parameters():
            param.requires_grad = False

        # Replace the fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  
        )
        return model

    def forward(self, x):
        return self.model(x)