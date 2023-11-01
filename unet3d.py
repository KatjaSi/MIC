import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Define the encoder (downsampling) path
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Define the middle (bottleneck) layer
        self.middle = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        # Define the decoder (upsampling) path
        self.decoder = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        # Encoder path
        x1 = self.encoder(x)
        
        # Middle (bottleneck) layer
        x2 = self.middle(x1)
        
        # Decoder path
        x3 = self.decoder(torch.cat((x1, x2), dim=1))
        
        return x3

# Define the input and output channels (based on your task)
in_channels = 4  # Assuming 4 input channels (e.g., different MRI modalities)
out_channels = 3  # Assuming 3 output channels (for tumor classes)

# Create the UNet3D model
model = UNet3D(in_channels, out_channels)

# Move the model to the device (e.g., GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Print the model architecture
print(model)
