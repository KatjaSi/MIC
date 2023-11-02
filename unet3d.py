import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3): # TODO: why input channels = 4?
        super(UNet3D, self).__init__()
        
        # Define the encoder (downsampling) path
        self.encoder = nn.Sequential(
           # nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
           # nn.BatchNorm3d(64),
           # nn.ReLU(inplace=True),
            Conv3DBlock(in_channels=in_channels, out_channels=32),
            Conv3DBlock(in_channels=32, out_channels=64),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv3DBlock(in_channels=64, out_channels=64),
            Conv3DBlock(in_channels=64, out_channels=128),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv3DBlock(in_channels=128, out_channels=128),
            Conv3DBlock(in_channels=128, out_channels=256),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Define the middle (bottleneck) layer
        self.middle = nn.Sequential(
           # nn.Conv3d(64, 128, kernel_size=3, padding=1),
           # nn.BatchNorm3d(128),
           # nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
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


class Conv3DBlock(nn.Module):
     
    def __init__(self, in_channels, out_channels):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))




# Define the input and output channels (based on your task)
#in_channels = 4  # Assuming 4 input channels (e.g., different MRI modalities)
#out_channels = 3  # Assuming 3 output channels (for tumor classes)

# Create the UNet3D model
#model = UNet3D(in_channels, out_channels)

# Move the model to the device (e.g., GPU)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = model.to(device)

# Print the model architecture
#print(model)
