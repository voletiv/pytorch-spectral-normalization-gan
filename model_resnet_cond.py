# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import numpy as np

channels = 3


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, cond_dim, emb_or_lin='emb'):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.cond_dim = cond_dim
        self.emb_or_lin = emb_or_lin
        if self.num_groups == 0:
            self.num_groups = self.num_features
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_features, affine=False)
        if self.emb_or_lin == 'emb':
            self.embed = nn.Embedding(self.cond_dim, self.num_features * 2)
            self.embed.weight.data[:, :self.num_features].fill_(1.)  # Initialize scale to 1
            self.embed.weight.data[:, self.num_features:].zero_()  # Initialize bias at 0
        else:
            self.embed = nn.Linear(self.cond_dim, self.num_features*2)

    def forward(self, x, y):
        out = self.group_norm(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ConditionalBatchNorm(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, cond_dim, emb_or_lin='emb'):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.bn = nn.BatchNorm2d(self.num_features, momentum=0.001, affine=False)
        self.emb_or_lin = emb_or_lin
        if self.emb_or_lin == 'emb':
            self.embed = nn.Embedding(self.cond_dim, self.num_features * 2)
            self.embed.weight.data[:, :self.num_features].fill_(1.)  # Initialize scale to 1
            self.embed.weight.data[:, self.num_features:].zero_()  # Initialize bias at 0
        else:
            self.embed = nn.Linear(self.cond_dim, self.num_features*2)

    def forward(self, x, condition):
        out = self.bn(x)
        gamma, beta = self.embed(condition).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_type="batch", num_classes=10):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if norm_type == "batch":
            self.norm1 = ConditionalBatchNorm(in_channels, num_classes, emb_or_lin='emb')
            self.norm2 = ConditionalBatchNorm(out_channels, num_classes, emb_or_lin='emb')
        elif norm_type == "group":
            self.norm1 = ConditionalGroupNorm(4, in_channels, num_classes, emb_or_lin='emb')
            self.norm2 = ConditionalGroupNorm(4, out_channels, num_classes, emb_or_lin='emb')

        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(in_channels),
        #     self.norm1,
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2),
        #     self.conv1,
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     self.conv2
        #     )

        self.model1 = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1
        )

        self.model2 = nn.Sequential(
            nn.ReLU(),
            self.conv2
        )

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x, cond):
        x1 = self.norm1(x, cond)
        x1 = self.model1(x1)
        x1 = self.norm2(x1, cond)
        x1 = self.model2(x1)

        return x1 + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128


class Generator(nn.Module):
    def __init__(self, z_dim, norm_type="batch", num_classes=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.norm_type = norm_type
        self.num_classes = num_classes

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        # self.model = nn.Sequential(
        #     ResBlockGenerator(GEN_SIZE, GEN_SIZE, 2, self.norm_type, self.num_classes),
        #     ResBlockGenerator(GEN_SIZE, GEN_SIZE, 2, self.norm_type, self.num_classes),
        #     ResBlockGenerator(GEN_SIZE, GEN_SIZE, 2, self.norm_type, self.num_classes),
        #     nn.BatchNorm2d(GEN_SIZE),
        #     nn.ReLU(),
        #     self.final,
        #     nn.Tanh())

        self.block1 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, 2, self.norm_type, self.num_classes)
        self.block2 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, 2, self.norm_type, self.num_classes)
        self.block3 = ResBlockGenerator(GEN_SIZE, GEN_SIZE, 2, self.norm_type, self.num_classes)

        self.model = nn.Sequential(
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z, cond):
        x = self.dense(z).view(-1, GEN_SIZE, 4, 4)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

        self.cond_emb = nn.Embedding(num_classes, DISC_SIZE)
        nn.init.xavier_uniform_(self.cond_emb.weight.data, 1.)
        self.cond_emb = SpectralNorm(self.cond_emb)

    def forward(self, x, cond):
        disc_feat = self.model(x).view(-1, DISC_SIZE)
        output1 = self.fc(disc_feat)
        output2 = torch.sum(torch.mul(disc_feat, self.cond_emb(cond)), dim=1, keepdim=True)
        return output1 + output2
