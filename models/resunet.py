from torch import nn
import torch
from .layers import ResidualBlock

class ResUnet(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(ResUnet, self).__init__()
        self.encoder = ResUnetEncoder(input_depth_0 + input_depth_1, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        #concatenate sources
        x = torch.cat((x[0], x[1]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x

class JointFusion(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(JointFusion, self).__init__()
        
        self.encoder_0 = ResUnetEncoderNoSkipp(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoderNoSkipp(input_depth_1, depths)
        self.decoder = ResUnetDecoderNoSkipp(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        #concatenate sources
        x_0 = self.encoder_0(x[0])
        x_1 = self.encoder_1(x[1])
        x = torch.cat((x_0, x_1), dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class LateFusion(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(LateFusion, self).__init__()
        
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder_0 = ResUnetDecoder(depths)
        self.decoder_1 = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        #concatenate sources
        x_0 = self.encoder_0(x[0])
        x_1 = self.encoder_1(x[1])

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=1)
        x = self.classifier(x)
        return x

class ResUnetOpt(nn.Module):
    def __init__(self, input_depth, depths, n_classes):
        super(ResUnetOpt, self).__init__()
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        x = x[0]

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x

class ResUnetEncoderNoSkipp(nn.Module):
    def __init__(self, input_depth, depths):
        super(ResUnetEncoderNoSkipp, self).__init__()
        self.first_res_block = nn.Sequential(
            nn.Conv2d(input_depth, depths[0], kernel_size=3, padding=1, padding_mode = 'reflect'),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], kernel_size=3, padding=1, padding_mode = 'reflect')
        )
        self.first_res_cov = nn.Conv2d(input_depth, depths[0], kernel_size=1)

        self.enc_block_0 = ResidualBlock(depths[0], depths[1], stride = 2)
        self.enc_block_1 = ResidualBlock(depths[1], depths[2], stride = 2)
        self.enc_block_2 = ResidualBlock(depths[2], depths[3], stride = 2)

    def forward(self, x):
        #first block
        x_idt = self.first_res_cov(x)
        x = self.first_res_block(x)
        x_0 = x + x_idt

        #encoder blocks
        x_1 = self.enc_block_0(x_0)
        x_2 = self.enc_block_1(x_1)
        x_3 = self.enc_block_2(x_2)

        return x_3

class ResUnetEncoder(nn.Module):
    def __init__(self, input_depth, depths):
        super(ResUnetEncoder, self).__init__()
        self.first_res_block = nn.Sequential(
            nn.Conv2d(input_depth, depths[0], kernel_size=3, padding=1, padding_mode = 'reflect'),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], kernel_size=3, padding=1, padding_mode = 'reflect')
        )
        self.first_res_cov = nn.Conv2d(input_depth, depths[0], kernel_size=1)

        self.enc_block_0 = ResidualBlock(depths[0], depths[1], stride = 2)
        self.enc_block_1 = ResidualBlock(depths[1], depths[2], stride = 2)
        self.enc_block_2 = ResidualBlock(depths[2], depths[3], stride = 2)

    def forward(self, x):
        #first block
        x_idt = self.first_res_cov(x)
        x = self.first_res_block(x)
        x_0 = x + x_idt

        #encoder blocks
        x_1 = self.enc_block_0(x_0)
        x_2 = self.enc_block_1(x_1)
        x_3 = self.enc_block_2(x_2)

        return x_0, x_1, x_2, x_3

class ResUnetDecoderNoSkipp(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoderNoSkipp, self).__init__()
        self.dec_block_2 = ResidualBlock(2*depths[3], depths[2])
        self.dec_block_1 = ResidualBlock(depths[2], depths[1])
        self.dec_block_0 = ResidualBlock(depths[1], depths[0])

        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_0 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x
        #concatenate sources
        x_2u = self.upsample_2(x)
        x_2 = self.dec_block_2(x_2u)

        x_1u = self.upsample_1(x_2)
        x_1 = self.dec_block_1(x_1u)

        x_0u = self.upsample_0(x_1)
        x_0 = self.dec_block_0(x_0u)

        return x_0

class ResUnetDecoder(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoder, self).__init__()
        self.dec_block_2 = ResidualBlock(depths[2] + depths[3], depths[2])
        self.dec_block_1 = ResidualBlock(depths[1] + depths[2], depths[1])
        self.dec_block_0 = ResidualBlock(depths[0] + depths[1], depths[0])

        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_0 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = x
        #concatenate sources
        x_2u = self.upsample_2(x_3)
        x_2c = torch.cat((x_2u, x_2), dim=1)
        x_2 = self.dec_block_2(x_2c)

        x_1u = self.upsample_1(x_2)
        x_1c = torch.cat((x_1u, x_1), dim=1)
        x_1 = self.dec_block_1(x_1c)

        x_0u = self.upsample_0(x_1)
        x_0c = torch.cat((x_0u, x_0), dim=1)
        x_0 = self.dec_block_0(x_0c)

        return x_0

class ResUnetClassifier(nn.Module):
    def __init__(self, depth, n_classes):
        super(ResUnetClassifier, self).__init__()
        self.res_block = ResidualBlock(depth, depth)
        self.last_conv = nn.Conv2d(depth, n_classes, kernel_size=1)
        self.last_act = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.res_block(x)
        x = self.last_conv(x)
        x = self.last_act(x)
        return x
