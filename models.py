from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_connection = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, stride=1, bias=False,
        )
        self.batchnorm_connection = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        batch_size, in_channels, in_height, in_width = input.shape
        assert self.in_channels == in_channels

        identity: torch.Tensor = input

        # Local linear transformation
        output: torch.Tensor = self.conv1(input)
        assert output.shape == (batch_size, self.out_channels, in_height, in_width)
        output = self.batchnorm1(output)
        output = F.relu(output)

        output = self.conv2(output)
        assert output.shape == (batch_size, self.out_channels, in_height, in_width)
        output = self.batchnorm2(output)

        # Skip Connection
        identity = self.conv_connection(identity)
        identity = self.batchnorm_connection(identity)

        # Merge
        assert output.shape == identity.shape
        output: torch.Tensor = output + identity
        output = F.relu(output)
        
        assert output.shape == (batch_size, self.out_channels, in_height, in_width)
        return output


class Encoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = UnitBlock(in_channels=1, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UnitBlock(in_channels=64, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UnitBlock(in_channels=128, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UnitBlock(in_channels=256, out_channels=512)


    def forward(self, input: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        enc1: torch.Tensor = self.encoder1(input)
        enc2: torch.Tensor = self.encoder2(self.pool1(enc1))
        enc3: torch.Tensor = self.encoder3(self.pool2(enc2))
        enc4: torch.Tensor = self.encoder4(self.pool3(enc3))
        return enc1, enc2, enc3, enc4
    

class BottleNeck(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UnitBlock(in_channels=512, out_channels=1024)
        self.upconv = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, 
            kernel_size=2, stride=2, bias=False,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        batch_size, in_channels, height, width = input.shape
        assert in_channels == 512
        output: torch.Tensor = self.pool(input)
        output: torch.Tensor = self.bottleneck(output)
        output: torch.Tensor = self.upconv(output)
        assert output.shape == (batch_size, 512, height, width)
        return output


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder1 = UnitBlock(in_channels=1024, out_channels=512)
        self.upconv1 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, 
            kernel_size=2, stride=2, bias=False,
        )
        self.decoder2 = UnitBlock(in_channels=512, out_channels=256)
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, 
            kernel_size=2, stride=2, bias=False,
        )
        self.decoder3 = UnitBlock(in_channels=256, out_channels=128)
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, 
            kernel_size=2, stride=2, bias=False, 
        )
        self.decoder4 = UnitBlock(in_channels=128, out_channels=64)
        self.prediction_head = nn.Conv2d(
            in_channels=64, out_channels=1, 
            kernel_size=1, stride=1, bias=False,
        )
    
    def forward(
        self,
        input: torch.Tensor, 
        encoder_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    ) -> torch.Tensor:
        assert input.ndim == 4
        batch_size, in_channels, in_height, in_width = input.shape
        assert in_channels == 512
        enc1, enc2, enc3, enc4 = encoder_outputs

        assert enc4.shape == input.shape
        output: torch.Tensor = self.decoder1(torch.cat(tensors=[enc4, input], dim=1))
        output: torch.Tensor = self.upconv1(output)

        assert enc3.shape == output.shape
        output: torch.Tensor = self.decoder2(torch.cat(tensors=[enc3, output], dim=1))
        output: torch.Tensor = self.upconv2(output)

        assert enc2.shape == output.shape
        output: torch.Tensor = self.decoder3(torch.cat(tensors=[enc2, output], dim=1))
        output: torch.Tensor = self.upconv3(output)

        assert enc1.shape == output.shape
        output: torch.Tensor = self.decoder4(torch.cat(tensors=[enc1, output], dim=1))
        output: torch.Tensor = self.prediction_head(output)

        return output
    

class UNet(nn.Module):

    def __init__(
        self, 
        encoder: Encoder, 
        bottleneck: BottleNeck, 
        decoder: Decoder = 64, 
    ) -> None:
        
        super().__init__()
        self.encoder: Encoder = encoder
        self.bottleneck: BottleNeck = bottleneck
        self.decoder: Decoder = decoder

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        batch_size, in_channels, in_height, in_width = input.shape
        enc1, enc2, enc3, enc4 = self.encoder(input)
        output: torch.Tensor = self.bottleneck(enc4)
        output: torch.Tensor = self.decoder(input=output, encoder_outputs=[enc1, enc2, enc3, enc4])
        assert output.shape == (batch_size, 1, in_height, in_width)
        return output



# TEST
if __name__ == '__main__':
    self = UNet(encoder=Encoder(), bottleneck=BottleNeck(), decoder=Decoder())
    x = torch.rand(32, 1, 128, 512)
    y = self(x)
    print(x.shape)
    print(y.shape)




        