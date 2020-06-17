# https://github.com/voletiv/inception-score-pytorch
# https://github.com/voletiv/pytorch-fid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import imageio
import numpy as np
import os
import subprocess
import time
import torch
import tqdm

import numpy as np
import os
import pathlib
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from imageio import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models
from torch.nn.functional import interpolate

from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

from scipy.stats import entropy

import model_resnet_cond

# from calc_IS_FID_CAS import *; run_me('batch', '/home/voletivi/scratch/sngan_christiancosgrove_cifar10/CBN2/checkpoints', '/home/voletivi/scratch/Datasets/CIFAR10_FID_ref_mean_std_per_class.npz', '/home/voletivi/scratch/Datasets/CIFAR10')


def run_me(cond, checkpoints_dir, cifar10_data_path, norm_type='batch', gen_size=256, FID_ref_per_class_npz_path=None,
            n_samples_per_class=5000, Z_dim=128, num_classes=10, epoch_freq=1, batch_size=256):
    # FID
    if FID_ref_per_class_npz_path is None:
        FID_ref_m, FID_ref_s = calc_for_CIFAR10(cifar10_data_path)
    else:
        a = np.load(FID_ref_per_class_npz_path)
        FID_ref_m = a['mean']
        FID_ref_s = a['std']
    # Generator
    if cond:
        G = model_resnet_cond.Generator(Z_dim, norm_type=norm_type, num_classes=num_classes).cuda()
    else:
        G = model_resnet.Generator(Z_dim, gen_size).cuda()
    # Checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(checkpoints_dir, "gen_*")))
    epochs = np.array(sorted([int(os.path.basename(p).split('_')[-1]) for p in ckpt_files]))[::epoch_freq]
    # epochs = [1, 6, 11, 15, 20, 30, 40, 50]
    file_prefix = os.path.join(os.path.dirname(ckpt_files[0]), os.path.basename(ckpt_files[0]).split('_')[0])
    # gen_samples_path = os.path.join(checkpoints_dir, '../gen_samples')
    # is_npz_save_path = os.path.join(checkpoints_dir, '../is.npz')
    # fid_npz_save_path = os.path.join(checkpoints_dir, '../fid.npz')
    save_path = os.path.realpath(os.path.join(checkpoints_dir, '..'))
    CAS_npz_path = os.path.realpath(os.path.join(checkpoints_dir, '..', 'CAS_acc.npz'))
    if os.path.exists(CAS_npz_path):
        a = np.load(CAS_npz_path)
        ckpts, CAS_acc = list(a['ckpts']), list(a['acc'])
    else:
        ckpts, CAS_acc = [], []
    # For each epoch
    for epoch in epochs:
        print("Epoch", epoch)
        # if epoch in ckpts:
        #     continue
        # Load
        pth_filename = file_prefix + '_' + str(epoch)
        G.load_state_dict(torch.load(pth_filename))
        print("Generating", n_samples_per_class, "samples per class, for", num_classes, "classes")
        # Generate images
        images_per_class = generate_n_samples_per_class(G, n_samples_per_class=n_samples_per_class, batch_size=batch_size, save=False, num_of_classes=num_classes)
        # Calc IS, FID
        calc_IS_FID_and_save(images_per_class, epoch, save_path, ref_m_per_class=FID_ref_m, ref_s_per_class=FID_ref_s, num_of_classes=num_classes)
        # Calc CAS
        # print("Calculating CAS")
        # CAS(CAS_npz_path, epoch, images_per_class, cifar10_data_path)
        # # Delete gen_samples dir
        # subprocess.run("nvidia-smi")
        # print("Deleting gen_samples_dir")
        # shutil.rmtree(os.path.join(save_path, 'gen_samples'))
        # print("Done")


def generate_n_samples_per_class(G, n_samples_per_class=256,
                                 save=False, gen_samples_path='gen_samples',
                                 batch_size=64, noise_dim=128, imsize=32, device=torch.device('cuda'),
                                 # G_args={'q_tokens':None, 'a_tokens':None},
                                 separate_class_dir=False, num_of_classes=10):
    # Sample n_samples_per_class # of images from generator, and save in 'gen_samples'
    images_per_class = []
    G.eval()
    with torch.set_grad_enabled(False):
        for class_label in tqdm.tqdm(range(num_of_classes)):
            images = torch.empty((0, 3, imsize, imsize))
            print("Generating samples for class", class_label)
            for i in tqdm.tqdm(range((n_samples_per_class - 1)//batch_size + 1)):
                z = torch.randn(min(batch_size, n_samples_per_class - i*batch_size), noise_dim).to(device)
                l = torch.tensor([class_label]*len(z)).to(device)
                # ims = G(z, l, **G_args).detach().cpu().add(1.).div(2.).mul(255.).numpy().transpose(0, 2, 3, 1).astype('uint8')
                # ims = G(z, l).detach().cpu().add(1.).div(2.).mul(255.).numpy().transpose(0, 2, 3, 1).astype('uint8')
                ims = G(z, l).detach().cpu()
                # import pdb; pdb.set_trace()
                images = torch.cat((images, ims))
            if save:
                # Save all in folder
                print("Saving samples")
                if separate_class_dir:
                    class_samples_dir = os.path.join(gen_samples_path, str(class_label), 'blah')
                else:
                    class_samples_dir = os.path.join(gen_samples_path, str(class_label))
                os.makedirs(class_samples_dir)
                for i in tqdm.tqdm(range(n_samples_per_class)):
                    imageio.imwrite(os.path.join(class_samples_dir, '{:04d}.png'.format(i)), images[i])
            else:
                images_per_class.append(images)
    # Return
    G.train()
    if not save:
        return images_per_class


def CAS(npz_save_path, epoch, images_per_class, cifar10_data_path, batch_size=200):
    if os.path.exists(npz_save_path):
        a = np.load(npz_save_path)
        ckpts, CAS_acc = list(a['ckpts']), list(a['acc'])
    else:
        ckpts, CAS_acc = [], []
    # Append to ckpts
    if epoch in ckpts:
            return
    ckpts.append(epoch)
    # Classifier
    train_dataset = torch.utils.data.TensorDataset(torch.cat(images_per_class), torch.tensor([[i]*len(images_per_class[i]) for i in range(len(images_per_class))]).view(-1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    valid_dataset = dset.CIFAR10(root=cifar10_data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2 - 1)]))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    dataloaders = {"train": train_loader, "val": valid_loader}
    save_dir = os.path.join(os.path.dirname(npz_save_path), f'cifar10_classifier_resnet18_bs200_ckpt_{epoch:07d}')
    best_acc = CAS_train_model(dataloaders, save_dir=save_dir, num_iters=2000, val_every_n_iters=50)
    del train_dataset, valid_dataset, dataloaders
    CAS_acc.append(best_acc)
    np.savez(npz_save_path, ckpts=ckpts, acc=CAS_acc)


# https://www.kaggle.com/gntoni/using-pytorch-resnet
def CAS_train_model(dataloaders, save_dir="./cifar10_classifier", device=torch.device('cuda'),
                    criterion=nn.CrossEntropyLoss(), lr=0.0001, num_iters=100000, val_every_n_iters=1):
    def get_samples(data_iter, device, dataloader, scale_factor=224./32.):
        try:
            inputs, labels = next(data_iter)
        except:
            data_iter = iter(dataloader)
            inputs, labels = next(data_iter)
        # Then
        inputs, labels = interpolate(inputs, scale_factor=scale_factor).to(device), labels.to(device)
        return inputs, labels, data_iter

    def make_plot(save_dir, train_losses, train_accs, val_losses, val_accs, val_every_n_iters):
        # Plot
        iters_x = np.arange(len(train_losses))*val_every_n_iters
        fig = plt.figure(figsize=(10, 20))
        plt.subplot(211)
        plt.plot(iters_x, np.zeros(iters_x.shape), 'k--', alpha=0.5)
        plt.plot(iters_x, train_losses, color='C1', alpha=0.7, label='train_loss')
        plt.plot(iters_x, val_losses, color='C2', alpha=0.7, label='val_loss')
        plt.legend()
        if max(train_losses) > 2 or max(val_losses) > 2:
            plt.yscale("symlog")
        plt.title("Loss")
        plt.xlabel("Iterations")
        plt.subplot(212)
        plt.plot(iters_x, train_accs, color='C1', alpha=0.7, label='train_acc')
        plt.plot(iters_x, val_accs, color='C2', alpha=0.7, label='val_acc')
        plt.legend()
        plt.ylim([0, 1])
        plt.title("Accuracy")
        plt.xlabel("Iterations")
        plt.savefig(os.path.join(save_dir, "plots.png"), bbox_inches='tight', pad_inches=0.5)
        plt.clf()
        plt.close()

    model = models.resnet18(pretrained=False, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Copy
    # shutil.copy(os.path.realpath('cifar10_classifier.py'), save_dir)
    try:
        since = time.time()
        best_model_wts = model.state_dict()
        best_acc = 0.0
        train_data_iter = iter(dataloaders['train'])
        val_data_iter = iter(dataloaders['val'])
        running_loss = []
        running_acc = []
        losses = []
        accs = []
        val_losses = []
        val_accs = []
        # For epochs
        model.train(True)
        print("About to train")
        for iter_num in range(num_iters):
            # print('-' * 10)
            # TRAIN
            inputs, labels, train_data_iter = get_samples(train_data_iter, device, dataloaders['train'])
            if iter_num == 0:
                print("got the samples")
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            if iter_num == 0:
                print("got the model output")
            loss = criterion(outputs, labels)
            if iter_num == 0:
                print("got the loss")
            # backward + optimize only if in training phase
            loss.backward()
            if iter_num == 0:
                print("did backward")
            optimizer.step()
            if iter_num == 0:
                print("did optimizer step")
            # statistics
            # import pdb; pdb.set_trace()
            running_loss.append(loss.item())
            _, preds = torch.max(outputs, 1)
            running_acc.append(torch.sum(preds.cpu() == labels.cpu()).float().item()/dataloaders['train'].batch_size)
            print("Iter {}/{}".format(iter_num+1, num_iters), "; Loss", "{:.04f}".format(np.mean(running_loss)), "; Acc", "{:.04f}".format(np.mean(running_acc)))
            del outputs, loss
            # VAL
            if iter_num % val_every_n_iters == 0:
                losses.append(np.mean(running_loss))
                accs.append(np.mean(running_acc))
                time_elapsed = time.time() - since
                print('{:.0f}hr {:.0f}min {:.0f}secs'.format((time_elapsed/60//24)%24, (time_elapsed//60)%60, time_elapsed%60))
                print('TRAIN: Loss: {:.4f}; Acc {:.4f}'.format(losses[-1], accs[-1]))
                running_loss = []
                running_acc = []
                # Validate
                model.train(False)
                running_val_losses = []
                running_val_accs = []
                print("About to validate!")
                val_data_iter = iter(dataloaders['val'])
                for i in tqdm.tqdm(range(len(val_data_iter))):
                    inputs, labels, val_data_iter = get_samples(val_data_iter, device, dataloaders['val'])
                    outputs = model(inputs)
                    running_val_losses.append(criterion(outputs, labels).item())
                    _, preds = torch.max(outputs, 1)
                    preds = preds.cpu()
                    labels = labels.cpu()
                    running_val_accs.append(torch.sum(preds == labels).float().item()/len(inputs))
                    del preds, outputs
                # Print
                val_losses.append(np.mean(running_val_losses))
                val_accs.append(np.mean(running_val_accs))
                print("making plot")
                make_plot(save_dir, losses, accs, val_losses, val_accs, val_every_n_iters)
                print('VALID: Loss: {:.4f}; Acc {:.4f}'.format(val_losses[-1], val_accs[-1]))
                if val_accs[-1] >= best_acc:
                    print("Best model yet...")
                    best_acc = val_accs[-1]
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                # Early stopping
                if len(val_losses) > 10:
                    if val_losses[-1] >= min(val_losses[-4:-1]):
                        break
                # Back to train
                model.train(True)
                # subprocess.run(['nvidia-smi'])
    # Except
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    # End of training
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # Print acc
    print('Best val Acc: {:4f}'.format(best_acc))
    # # load best model weights
    # model.load_state_dict(best_model_wts)
    os.rename(os.path.join(save_dir, "best_model.pth"), os.path.join(save_dir, "best_model_valAcc{:0.04f}.pth".format(best_acc)))
    return best_acc


def calc_IS_FID_for_CIFAR10(cifar10_data_path='/home/voletivi/scratch/Datasets/CIFAR10', per_class=False, check=False):
    import torch
    import tqdm
    from torchvision import datasets, transforms
    loader = torch.utils.data.DataLoader(datasets.CIFAR10(cifar10_data_path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])), batch_size=1000, shuffle=False, num_workers=0, pin_memory=True)
    imgs = []
    y = []
    print("Generating images")
    for im, yy in tqdm.tqdm(loader):
        imgs.append(im)
        y.append(yy)
        if check:
            break
    # Concat
    imgs = torch.cat(imgs, dim=0)
    y = torch.cat(y, dim=0)
    # Inception score
    inception_model = None
    if per_class:
        # Per class
        print("IS per class")
        IS_mc, IS_sc = [], []
        for c in tqdm.tqdm(range(10)):
            image_set = imgs[y == c]
            mc, sc, inception_model = inception_score(image_set, gpu='0', batch_size=64, inception_model=inception_model, return_model=True)
            IS_mc.append(mc)
            IS_sc.append(sc)
    else:
        # Overall
        print("IS")
        IS_m, IS_s, inception_model = inception_score(imgs, gpu='0', batch_size=64, inception_model=inception_model, return_model=True)
    # FID mean and std
    FID_model = None
    if per_class:
        # Per class
        print("FID per class")
        FID_mc, FID_sc = [], []
        for c in tqdm.tqdm(range(10)):
            image_set = imgs[y == c]
            FID_model, mc, sc = calculate_fid_given_images_sets([image_set, None], batch_size=64, gpu='0', dims=2048, model=FID_model, return_only_model_m1_s1=True)
            FID_mc.append(mc)
            FID_sc.append(sc)
    else:
        # Overall
        print("FID")
        FID_model, FID_m, FID_s = calculate_fid_given_images_sets([imgs, None], batch_size=64, gpu='0', dims=2048, model=FID_model, return_only_model_m1_s1=True)
    # Return
    if per_class:
        return IS_mc, IS_sc, FID_mc, FID_sc
    else:
        return IS_m, IS_s, FID_m, FID_s


def calc_IS_FID_and_save(images_per_class, ckpt, save_path, n_samples_per_class=256,
                         inception_model=None, fid_model=None, ref_m_per_class=None, ref_s_per_class=None,
                         IS_gpu='0', FID_gpu='0', num_of_classes=10):
    # Generate samples
    # print("Generating", n_samples_per_class, "samples per class, for", num_of_classes, "classes")
    # generate_n_samples_per_class(G, config, n_samples_per_class, os.path.join(save_path, 'gen_samples'),
    #                              G_args=G_args, separate_class_dir=True)
    # Calculate inception score per class
    print("\nCalculating IS\n")
    # subprocess.run("nvidia-smi")
    for class_label in tqdm.tqdm(range(num_of_classes)):
        print("\n Calculating IS class", class_label, "\n")
        inception_model = calc_inception_score_and_save(ckpt, images_per_class[class_label],
                                                        os.path.join(save_path, 'IS_class{:02d}.npz'.format(class_label)),
                                                        inception_model, IS_gpu)
        # subprocess.run("nvidia-smi")
    del inception_model
    # Calculate FID per class
    print("\nCalculating FID\n")
    # subprocess.run("nvidia-smi")
    if ref_m_per_class is None:
        print("Will save FID_ref_mean_std_per_class")
        save_ref = True
        ref_m_per_class, ref_s_per_class = [], []
        for _ in num_of_classes:
            ref_m_per_class.append(None)
            ref_s_per_class.append(None)
    else:
        save_ref = False
    # For each class
    for class_label in tqdm.tqdm(range(num_of_classes)):
        print("\n", class_label, "\n")
        # FID
        fid_model, ref_m_per_class[class_label], ref_s_per_class[class_label] = calc_fid_and_save(ckpt,
                                                    images_per_class[class_label],
                                                    os.path.join(save_path, 'FID_class{:02d}.npz'.format(class_label)),
                                                    ref_m_per_class[class_label], ref_s_per_class[class_label],
                                                    fid_model, FID_gpu)
        # subprocess.run("nvidia-smi")
    del fid_model
    if save_ref:
        print("Saving FID ref_m_per_class, ref_s_per_class")
        np.savez(os.path.join(save_path, "FID_ref_mean_std_per_class.npz"), mean=ref_m_per_class, std=ref_s_per_class)
        return ref_m_per_class, ref_s_per_class


def calc_inception_score_and_save(ckpt, images, npz_save_path, inception_model=None, gpu='0'):
    # Read npz if it exists
    if os.path.exists(npz_save_path):
        a = np.load(npz_save_path)
        ckpts, mean, std = list(a['ckpts']), list(a['mean']), list(a['std'])
    else:
        ckpts, mean, std = [], [], []
    # Append to ckpts
    ckpts.append(ckpt)
    # Calc score
    m, s, inception_model = inception_score(images, gpu=gpu, batch_size=64,
                                            inception_model=inception_model, return_model=True)
    mean.append(m)
    std.append(s)
    np.savez(npz_save_path, ckpts=ckpts, mean=mean, std=std)
    # Return
    return inception_model


def calc_fid_and_save(ckpt, images, npz_save_path, ref_m, ref_s, model=None, gpu='0'):
    # Read npz if it exists
    if os.path.exists(npz_save_path):
        # print("Found existing npz")
        a = np.load(npz_save_path)
        ckpts, fids, ref_m, ref_s = list(a['ckpts']), list(a['fids']), a['ref_m'], a['ref_s']
    else:
        ckpts, fids = [], []
    # Append ckpt
    ckpts.append(ckpt)
    # If ref_m is not given, calculate that as well
    if ref_m is None:
        print("Calc FID: calc score for BOTH!!")
        fid, model, ref_m, ref_s = calculate_fid_given_images_sets([images, None], batch_size=64, gpu=gpu, dims=2048,
                                                             model=model, return_model=True,
                                                             calc_only_for_one_path=False, return_m2s2=True, m2=None, s2=None)
        np.savez(os.path.realpath(os.path.join(os.path.dirname(npz_save_path), 'FID_ref_mean_std.npz')),
                 mean=ref_m, std=ref_s)
    # ref_m and ref_s are given, only calc inception m and std on first path (and then compare)
    else:
        fid, model = calculate_fid_given_images_sets([images, None], batch_size=64, gpu=gpu, dims=2048,
                                               model=model, return_model=True,
                                               calc_only_for_one_path=True, m2=ref_m, s2=ref_s, return_m2s2=False)
    # Append, save
    fids.append(fid)
    np.savez(npz_save_path, ckpts=ckpts, fids=fids, ref_m=ref_m, ref_s=ref_s)
    # Return
    return model, ref_m, ref_s


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_activations(images, model, batch_size=64, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images       : Tensor of images NxCxWxH, in range (0, 1)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(images) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    n_batches = len(images) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm.tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        # images = np.array([imread(str(f)).astype(np.float32)
        #                    for f in files[start:end]])

        # # Reshape to (n_images, 3, height, width)
        # images = images.transpose((0, 3, 1, 2))[:, :3, :, :]
        # images /= 255

        # batch = torch.from_numpy(images).type(torch.FloatTensor)

        batch = images[start:end]

        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().detach().numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(images, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images       : Tensor of images NxCxWxH in range (-1, 1)
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images.add(1.).mul(.5), model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_images_sets(images_sets, batch_size, gpu='', dims=2048, return_only_model_m1_s1=False,
                              model=None, return_model=False,
                              calc_only_for_one_path=False, m2=None, s2=None, return_m2s2=False):
    """Calculates the FID of two paths"""
    """'paths' contain two directories, each of which contains images in their class_dirs"""
    """If 'calc_only_for_one_path' is True, then
           - only m1 and s1 are calculated for paths[0],
           - and m2 & s2 are given
    gpu: '0'
    """

    # for p in paths:
    #     if not os.path.exists(p):
    #         raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    if model is None:
        model = InceptionV3([block_idx])

    device = torch.device('cuda:' + gpu if gpu != '' else 'cpu')
    model = model.to(device)

    m1, s1 = calculate_activation_statistics(images_sets[0], model, batch_size, dims, gpu != '')

    if return_only_model_m1_s1:
        return model, m1, s1

    if not calc_only_for_one_path:
        m2, s2 = calculate_activation_statistics(images_sets[1], model, batch_size, dims, gpu != '')

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    if return_model and return_m2s2:
        return fid_value, model, m2, s2
    elif return_model and not return_m2s2:
        return fid_value, model
    elif not return_model and return_m2s2:
        return fid_value, m2, s2
    else:
        return fid_value


def inception_score(images, gpu='', batch_size=64, resize=True, splits=10,
                    inception_model=None, return_model=False):
    """Computes the inception score of the generated images imgs

    images -- Tensor of images in range (-1, 1)
    gpu -- id of GPU to be used (e.g. 0)
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    cuda = False if gpu == '' else True

    # imgs = dset.ImageFolder(root=data_path, transform=transforms.Compose([
    #                                                   transforms.ToTensor(),
    #                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    imgs = torch.utils.data.TensorDataset(images)

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=False)

    # Load inception model
    if inception_model is None:
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model = inception_model.to(torch.device('cuda:' + gpu if gpu != '' else 'cpu'))

    inception_model.eval();
    def get_pred(x):
        if resize:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).cpu().detach().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # import pdb; pdb.set_trace()
        batch = batch[0].type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in tqdm.tqdm(range(splits)):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    if return_model:
        return np.mean(split_scores), np.std(split_scores), inception_model
    else:
        return np.mean(split_scores), np.std(split_scores)
