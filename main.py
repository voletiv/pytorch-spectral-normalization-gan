import argparse
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import model_resnet
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# python main.py --model resnet --loss hinge --data_dir /home/voletivi/scratch/Datasets/CIFAR10 --out_dir /home/voletivi/scratch/sngan_christiancosgrove_cifar10/CGN5 --norm group

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./SNGAN')
parser.add_argument('--data_dir', type=str, default='/home/voletivi/scratch/Datasets/CIFAR10')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--begin_epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--disc_iters', type=int, default=1, help="number of updates to discriminator for every update to generator ")
parser.add_argument('--Z_dim', type=int, default=128)
parser.add_argument('--gen_size', type=int, default=256)
parser.add_argument('--disc_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--norm', choices=['batch', 'group'], default='batch')
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--samples_dir', type=str, default='samples')

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

args.checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
args.samples_dir = os.path.join(args.out_dir, args.samples_dir)

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
if args.model == 'resnet':
    discriminator = model_resnet.Discriminator(args.disc_size).cuda()
    generator = model_resnet.Generator(args.Z_dim, args.gen_size).cuda()
else:
    discriminator = model.Discriminator().cuda()
    generator = model.Generator(args.Z_dim).cuda()

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(args.disc_iters):
            z = Variable(torch.randn(args.batch_size, args.Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, args.Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 10 == 0:
            curr_time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            log = f'[{curr_time_str}] Epoch {epoch} (batch {batch_idx} of {len(loader)}) disc_loss {disc_loss.item():.04f} gen_loss {gen_loss.item():.04f}\n'
            logg(log)
    scheduler_d.step()
    scheduler_g.step()

fixed_z = Variable(torch.randn(args.batch_size, args.Z_dim).cuda())


def evaluate(epoch):
    generator.eval()
    samples = generator(fixed_z).cpu().data.numpy()[:64]
    generator.train()
    save_image(samples, os.path.join(args.samples_dir, f'{epoch:03d}.png'), nrow=10)

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.samples_dir, exist_ok=True)


def logg(log):
    print(log)
    log_file.write(log)
    log_file.flush()

# Logging
log_file = open(os.path.join(args.out_dir, 'log.txt'), "wt")
log = str(args) + '\n'
logg(log)

evaluate(0)
log = f"Saving disc_{0}, gen{0}\n"
logg(log)
torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(0)))
torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(0)))

for epoch in range(2000):
    train(epoch)
    evaluate(epoch)
    if epoch % args.save_freq == 0:
        log = f"Saving disc_{epoch}, gen_{epoch}\n"
        logg(log)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
        torch.save(optim_disc.state_dict(), os.path.join(args.checkpoint_dir, 'optim_disc_{}'.format(epoch)))
        torch.save(optim_gen.state_dict(), os.path.join(args.checkpoint_dir, 'optim_gen_{}'.format(epoch)))
