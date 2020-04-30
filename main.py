import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import model_resnet
import model

from calc_IS_FID_CAS import calc_IS_FID_for_CIFAR10, inception_score, calculate_fid_given_images_sets

# python main.py --model resnet --loss hinge --data_dir /home/voletivi/scratch/Datasets/CIFAR10 --out_dir /home/voletivi/scratch/sngan_christiancosgrove_cifar10/CGN5 --norm group

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./SNGAN')
parser.add_argument('--data_dir', type=str, default='/home/voletivi/scratch/Datasets/CIFAR10')
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--begin_epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--disc_iters', type=int, default=5, help="number of updates to discriminator for every update to generator ")
parser.add_argument('--Z_dim', type=int, default=128)
parser.add_argument('--gen_size', type=int, default=256)
parser.add_argument('--disc_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--norm', choices=['batch', 'group'], default='batch')
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--samples_dir', type=str, default='samples')
parser.add_argument('--isfid', type=eval, default=True)

parser.add_argument('--check', action='store_true', default='hinge')

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()
args.out_dir = os.path.join(os.path.dirname(args.out_dir), f'{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.path.basename(args.out_dir)}_bs{args.batch_size}_di{args.disc_iters}_z{args.Z_dim}_G{args.gen_size}_D{args.disc_size}_lr{args.lr}_beta_{args.beta1}_{args.beta2}')

args.checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
args.samples_dir = os.path.join(args.out_dir, args.samples_dir)
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.samples_dir, exist_ok=True)

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
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# use an exponentially decaying learning rate
# scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
# scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

# def adjust_lr(optimizer, epoch):
#     lr = args.lr * (0.1 ** (epoch // 20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def train(epoch):
    epoch_start_time = time.time()
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

        if batch_idx % 20 == 0:
            curr_time = time.time()
            curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
            elapsed = str(datetime.timedelta(seconds=(curr_time - epoch_start_time)))
            log = f'[{curr_time_str}] [{elapsed}] Epoch {epoch}, (batch {batch_idx} of {len(loader)}), disc_loss {disc_loss.item():.04f}, gen_loss {gen_loss.item():.04f}\n'
            logg(log)

        if args.check:
            break

    # scheduler_d.step()
    # scheduler_g.step()


def logg(log):
    print(log)
    log_file.write(log)
    log_file.flush()

# Logging
log_file = open(os.path.join(args.out_dir, 'log.txt'), "wt")
log = str(args) + '\n'
logg(log)

# EVALUATE
fixed_z = Variable(torch.randn(args.batch_size, args.Z_dim).cuda())


def evaluate(epoch):
    generator.eval()
    samples = generator(fixed_z).detach().cpu().add(1.).mul(0.5).data[:60]
    generator.train()
    save_image(samples, os.path.join(args.samples_dir, f'{epoch:03d}.png'), nrow=10)

evaluate(0)
log = f"Saving disc_{0}, gen{0}\n"
logg(log)
torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(0)))
torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(0)))

# IS, FID
if args.isfid:
    logg("Calculating for CIFAR10\n")
    IS_ref_m, IS_ref_s, FID_ref_m, FID_ref_s = calc_IS_FID_for_CIFAR10(args.data_dir, per_class=False, check=args.check)
    log = "CIFAR10 IS_m {IS_m}, IS_s {IS_s}, FID_ref_m {FID_ref_m}, FID_ref_s {FID_ref_s}"
    logg(log)


    def generate_samples(n_samples=50000):
        if args.check:
            n_samples = min(args.batch_size, 1000)
        generator.eval()
        with torch.set_grad_enabled(False):
            images = torch.empty((0, 3, 32, 32))
            for i in tqdm.tqdm(range((n_samples - 1)//args.batch_size + 1)):
                z = torch.randn(min(args.batch_size, n_samples - i*args.batch_size), args.Z_dim).cuda()
                ims = generator(z).detach().cpu()
                images = torch.cat((images, ims))
        generator.train()
        return images


    def plot_IS_FID(is_m, is_s, fids, inception_model, fid_model):
        images = generate_samples()
        logg("\nCalculating IS\n")
        m, s, inception_model = inception_score(images, gpu='0', batch_size=64, inception_model=inception_model, return_model=True)
        is_m.append(m)
        is_s.append(s)
        logg("\nCalculating FID\n")
        fid, fid_model = calculate_fid_given_images_sets([images, None], batch_size=64, gpu='0', dims=2048,
                                                   model=fid_model, return_model=True,
                                                   calc_only_for_one_path=True, m2=FID_ref_m, s2=FID_ref_s, return_m2s2=False)
        fids.append(fid)
        # Plot
        plt.errorbar(np.arange(len(fids))*args.save_freq, [IS_ref_m]*len(fids), yerr=[IS_ref_s]*len(fids), color='k', alpha=0.7, label="CIFAR10")
        plt.errorbar(np.arange(len(fids))*args.save_freq, is_m, yerr=is_s, alpha=0.7, label="IS")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "is.png"), bbox_inches='tight', pad_inches=0.1)
        plt.yscale("log")
        plt.savefig(os.path.join(args.out_dir, "is_log.png"), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        plt.plot(np.arange(len(fids))*args.save_freq, fids, alpha=0.7, label="FID")
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, "fid.png"), bbox_inches='tight', pad_inches=0.1)
        plt.yscale("log")
        plt.savefig(os.path.join(args.out_dir, "fid_log.png"), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return is_m, is_s, fids, inception_model, fid_model

    is_m, is_s, fids = [], [], []
    inception_model, fid_model = None, None
    is_m, is_s, fids, inception_model, fid_model = plot_IS_FID(is_m, is_s, fids, inception_model, fid_model)
    log = f"Epoch {0}, IS_mean {is_m[-1]}, IS_std {is_s[-1]}, FID {fids[-1]}\n"
    logg(log)

# TRAIN
start_time = time.time()
for epoch in range(args.begin_epoch, args.begin_epoch+args.n_epochs):
    train(epoch)
    # Time
    curr_time = time.time()
    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
    log = f"[{curr_time_str}] [{elapsed}] Epoch {epoch} train done.\n"
    logg(log)
    evaluate(epoch)
    # Time
    curr_time = time.time()
    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
    log = f"[{curr_time_str}] [{elapsed}] Epoch {epoch} eval done.\n"
    logg(log)
    if epoch % args.save_freq == 0:
        log = f"Saving disc_{epoch}, gen_{epoch}\n"
        logg(log)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
        torch.save(optim_disc.state_dict(), os.path.join(args.checkpoint_dir, 'optim_disc_{}'.format(epoch)))
        torch.save(optim_gen.state_dict(), os.path.join(args.checkpoint_dir, 'optim_gen_{}'.format(epoch)))
        # IS, FID
        if args.isfid:
            is_m, is_s, fids, inception_model, fid_model = plot_IS_FID(is_m, is_s, fids, inception_model, fid_model)
        # Time
        curr_time = time.time()
        curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
        log = f"[{curr_time_str}] [{elapsed}] Epoch {epoch}, IS_mean {is_m[-1]}, IS_std {is_s[-1]}, FID {fids[-1]}\n"
        logg(log)
