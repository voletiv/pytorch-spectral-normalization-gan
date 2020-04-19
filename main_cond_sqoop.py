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
import model_resnet_cond_sqoop
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from sqoop_dataloader import *

# python main_cond_sqoop.py --model resnet --loss hinge --out_dir /home/voletivi/scratch/sngan_christiancosgrove_cifar10/sqoop --data_dir /home/voletivi/scratch/Datasets/sqoop/sqoop_1obj_rep1000 --cond_mode obj --norm group

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--out_dir', type=str, default='./SNGAN')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--samples_dir', type=str, default='samples')
parser.add_argument('--norm', choices=['batch', 'group'], default='batch')
parser.add_argument('--sample_step', type=int, default=1)
parser.add_argument('--model_save_step', type=int, default=5)

parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--partition', type=str, default='train')
parser.add_argument('--buggy_dataset', action='store_true')

parser.add_argument('--cond_mode', type=str, choices=['obj', 'obj_obj', 'obj_rel_obj'])
parser.add_argument('--cond_dim', type=int, default=64)

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

args.checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
args.samples_dir = os.path.join(args.out_dir, args.samples_dir)

# loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(args.data_dir, train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
#         batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

ds = SqoopDataset(data_root=args.data_dir, partition=args.partition,
                  bug_of_above_vs_below=args.buggy_dataset)
# bs = args.batch_size_in_gpu//2 if args.use_DSloss else args.batch_size_in_gpu
loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=8, drop_last=True, pin_memory=True,
                    collate_fn=lambda batch: ds.sqoop_collate_fn_batch_read_feats(batch))

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
if args.model == 'resnet':
    discriminator = model_resnet_cond_sqoop.Discriminator(cond_mode=args.cond_mode, cond_dim=args.cond_dim).cuda()
    generator = model_resnet_cond_sqoop.Generator(Z_dim, cond_mode=args.cond_mode, norm_type=args.norm, cond_dim=args.cond_dim).cuda()
# else:
#     discriminator = model.Discriminator().cuda()
#     generator = model.Generator(Z_dim).cuda()

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)


def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        if args.cond_mode == 'obj':
            target = target[:, 0]
        elif args.cond_mode == 'obj_obj':
            target = target[:, [0, 2]]
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data, target)).mean() + nn.ReLU()(1.0 + discriminator(generator(z, target), target)).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data, target).mean() + discriminator(generator(z, target), target).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data, target), Variable(torch.ones(args.batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z, target), target), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()
        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z, target), target).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z, target), target), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()
        if batch_idx % 100 == 0:
            curr_time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{curr_time_str}] Epoch {epoch} (batch {batch_idx} of {len(loader)}) disc_loss {disc_loss.item():.04f} gen_loss {gen_loss.item():.04f}')
    scheduler_d.step()
    scheduler_g.step()

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
# fixed_cond = torch.from_numpy(np.tile(np.arange(10), args.batch_size//10 + 1)[:args.batch_size]).cuda()
_, fixed_cond = next(iter(loader))
if args.cond_mode == 'obj':
    fixed_cond = fixed_cond[:, 0]
elif args.cond_mode == 'obj_obj':
    fixed_cond = fixed_cond[:, [0, 2]]

fixed_cond = fixed_cond.cuda()


def evaluate(epoch):
    generator.eval()
    samples = generator(fixed_z, fixed_cond).detach().cpu().add(1.).mul(0.5).data[:60]
    generator.train()
    save_image(samples, os.path.join(args.samples_dir, f'{epoch:03d}.png'), nrow=10)

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.samples_dir, exist_ok=True)

evaluate(0)
torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(0)))
torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(0)))

for epoch in range(20000):
    train(epoch)
    if epoch % args.sample_step == 0:
        evaluate(epoch)
    if epoch % args.model_save_step == 0:
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, f'disc_{epoch:05d}'))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f'gen_{epoch:05d}'))
        torch.save(optim_disc.state_dict(), os.path.join(args.checkpoint_dir, f'optim_disc_{epoch:05d}'))
        torch.save(optim_gen.state_dict(), os.path.join(args.checkpoint_dir, f'optim_gen_{epoch:05d}'))
        torch.save(scheduler_d.state_dict(), os.path.join(args.checkpoint_dir, f'sch_disc_{epoch:05d}'))
        torch.save(scheduler_g.state_dict(), os.path.join(args.checkpoint_dir, f'sch_gen_{epoch:05d}'))
