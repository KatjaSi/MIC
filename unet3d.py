import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet3D,self).__init__()
        # Define the encoder (downsampling) path
        self.encoder = Encoder(in_channels=in_channels)
        
        # Define the middle (bottleneck) layer
        self.middle = nn.Sequential(
            Conv3DBlock(in_channels=256, out_channels=256),
            Conv3DBlock(in_channels=256, out_channels=512),
            UpConv3DBlock(in_channels=512, out_channels=512)
        )
        
        # Define the decoder (upsampling) path
        self.decoder = Decoder(in_channels=512, out_channels=out_channels) 
        

        
    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        x4 = self.middle(x3)
        x5 = self.decoder(x1,x2,x3,x4)
        return x5


class UpConv3DBlock(nn.Module):
     
    def __init__(self, in_channels, out_channels):
        super(UpConv3DBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(2,2,2), stride=2) # deconvolution
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Conv3DBlock(nn.Module):
     
    def __init__(self, in_channels, out_channels):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = Conv3DBlock(in_channels=in_channels, out_channels=32)
        self.conv2 = Conv3DBlock(in_channels=32, out_channels=64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = Conv3DBlock(in_channels=64, out_channels=64)
        self.conv4 = Conv3DBlock(in_channels=64, out_channels=128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5 = Conv3DBlock(in_channels=128, out_channels=128)
        self.conv6 = Conv3DBlock(in_channels=128, out_channels=256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.pool1(self.conv2(self.conv1(x)))
        x2 = self.pool2(self.conv4(self.conv3(x1)))
        x3 = self.pool3(self.conv6(self.conv5(x2)))
        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = Conv3DBlock(in_channels=in_channels, out_channels=512) 
        self.conv2 = Conv3DBlock(in_channels=256+512, out_channels=256)
        self.upconv1 = UpConv3DBlock(in_channels=256, out_channels=256)
        self.conv3 = Conv3DBlock(in_channels=128+256, out_channels=128) 
        self.conv4 = Conv3DBlock(in_channels=128, out_channels=128)
        self.upconv2 = UpConv3DBlock(in_channels=128, out_channels=128)
        self.conv5 = Conv3DBlock(in_channels=64+128, out_channels=64) 
        self.conv6 = Conv3DBlock(in_channels=64, out_channels=64)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self,x1,x2,x3,x4):
        x = self.conv1(x4) 
        x3_upsampled = nn.functional.interpolate(x3, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat((x3_upsampled, x), dim=1)
        x = self.conv2(x)

        x = self.upconv1(x)

        x2 = nn.functional.interpolate(x2, size=x.shape[2:], mode='trilinear', align_corners=False)
        x =torch.cat((x2,x), dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.upconv2(x)
        x1 = nn.functional.interpolate(x1, size=x.shape[2:], mode='trilinear', align_corners=False)
        x =torch.cat((x1,x), dim=1)
        x = self.conv5(x) # torch.cat(x1,x),dim=1
        x = self.conv6(x)
        x = self.final_conv(x)
        return x



# Define the input and output channels (based on your task)
#in_channels = 4  # Assuming 4 input channels (e.g., different MRI modalities)
#out_channels = 3  # Assuming 3 output channels (for tumor classes)

# Create the UNet3D model
model = UNet3D(in_channels=4, out_channels=3)

# Move the model to the device (e.g., GPU)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = model.to(device)

# Print the model architecture
print(model)

