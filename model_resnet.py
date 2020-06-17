# ResNet generator and discriminator
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import numpy as np


class ResBlockGenerator(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, sn=False):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        if sn:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)
            self.bypass_conv = SpectralNorm(self.bypass_conv)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            self.conv2
            )

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                self.bypass_conv
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, sn=True):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if sn:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            if sn:
                self.bypass_conv = SpectralNorm(self.bypass_conv)

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_ch == out_ch:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_ch,out_ch, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, sn=True):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        if sn:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)
            self.bypass_conv = SpectralNorm(self.bypass_conv)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, z_dim, GEN_SIZE=256, sn=False, im_ch=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.GEN_SIZE = GEN_SIZE
        self.sn = sn

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, im_ch, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)
        if sn:
            self.dense = SpectralNorm(self.dense)
            self.final = SpectralNorm(self.final)

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2, sn=sn),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2, sn=sn),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2, sn=sn),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.GEN_SIZE, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, DISC_SIZE=128, sn=True, im_ch=3):
        super(Discriminator, self).__init__()

        self.DISC_SIZE = DISC_SIZE
        self.sn = sn

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(im_ch, DISC_SIZE, stride=2, sn=sn),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2, sn=sn),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, sn=sn),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, sn=sn),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        if sn:
            self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, self.DISC_SIZE))