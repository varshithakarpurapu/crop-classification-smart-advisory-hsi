import torch.nn as nn
import torch.nn.functional as F

class FastHSIModel(nn.Module):
    def __init__(self, input_channels=200, num_classes=16, patch_size=11):
        super(FastHSIModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, (3,3,7), padding=(1,1,3))
        self.conv2 = nn.Conv3d(8, 16, (3,3,5), padding=(1,1,2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
