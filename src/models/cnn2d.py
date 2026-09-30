import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightAudioCNN2D(nn.Module):
    """
    Lightweight 2D ConvNet for Speech Audio Digit Classification.
    Input Tensor Shape: (Batch, 3, 40, T)
        - 3 Channels: MFCC (Static), Delta, Delta-Delta
        - 40 Bins: MFCC coefficients
        - T Time Frames: Spectral frames (~32 frames for 1.0s audio at 16kHz)
    Parameters: ~0.58M params
    """
    def __init__(self, num_classes=10, in_channels=3, n_mfcc=40):
        super(LightweightAudioCNN2D, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.drop3 = nn.Dropout(0.3)
        
        # FC Head
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Head
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        logits = self.fc2(x)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = LightweightAudioCNN2D()
    dummy_input = torch.randn(8, 3, 40, 32)
    output = model(dummy_input)
    print(f"Model created successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
