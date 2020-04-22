import glob
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

# INIT

# # ref_is_npz = '/scratch/voletivi/sagan/CIFAR10_IS_ref.npz'
# ref_is_npz = '/home/voletiv/EXPERIMENTS/sagan/CLEVR_PREV_RESULTS_NeurIPS_CIFAR10/CIFAR10_NeurIPS/CIFAR10_IS_ref.npz'
# ref_is = np.load(ref_is_npz)
# ref_is_mean = ref_is['mean']
# ref_is_std = ref_is['std']

ref_is_npz = '/home/voletiv/Datasets/CIFAR10/CIFAR10_IS_FID.npz'
ref_is = np.load(ref_is_npz)
ref_is_mean = ref_is['IS_mean']
ref_is_std = ref_is['IS_std']

CIFAR10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# labels = ['BN', 'LN', 'GN', 'IN']

# dirs_CBN = glob.glob('/scratch/voletivi/sagan/experiments/*_WGANGP_CIFAR10_CBN_CIFAR10/')
# dirs_CLN = glob.glob('/scratch/voletivi/sagan/experiments/*_WGANGP_CIFAR10_CLN_CIFAR10/')
# dirs_CGN = glob.glob('/scratch/voletivi/sagan/experiments/*_WGANGP_CIFAR10_CGN4_CIFAR10/')
# dirs_CIN = glob.glob('/scratch/voletivi/sagan/experiments/*_WGANGP_CIFAR10_CIN_CIFAR10/')

# dirs_CBN = sorted(glob.glob('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN*'))[1:][:3]
# dirs_CGN = sorted(glob.glob('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN*'))[:3]

# # DIRS
# dirs_CBN_bs64 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN2',
#                  '/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN3',
#                  '/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN4']
# dirs_CGN_bs64 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN',
#                  '/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN2',
#                  '/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN3']
# dirs_CBN_bs32 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN_bs32']
# dirs_CGN_bs32 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN_bs32']
# dirs_CBN_bs16 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN_bs16']
# dirs_CGN_bs16 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN_bs16']
# dirs_CBN_bs8 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN_bs8']
# dirs_CGN_bs8 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN_bs8']
# dirs_CBN_bs4 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CBN_bs4']
# dirs_CGN_bs4 = ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CGN_bs4']

# # dirs = [dirs_CBN, dirs_CLN, dirs_CGN, dirs_CIN]
# dirs = [dirs_CBN_bs64, dirs_CGN_bs64, dirs_CBN_bs32, dirs_CGN_bs32, dirs_CBN_bs16, dirs_CGN_bs16, dirs_CBN_bs8, dirs_CGN_bs8, dirs_CBN_bs4, dirs_CGN_bs4]
# labels = ['CBN_bs64', 'CGN_bs64', 'CBN_bs32', 'CGN_bs32', 'CBN_bs16', 'CGN_bs16', 'CBN_bs8', 'CGN_bs8', 'CBN_bs4', 'CGN_bs4']

dirs = [['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/exp_CBN'], ['/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/exp_CGN']]
labels = ['CBN', 'CGN']

#######################################################
# CAS

CAS_CBN_ckpts = []
CAS_CBN_acc = []
for d in dirs_CBN_bs64:
    a = np.load(os.path.join(d, 'CAS_acc.npz'))
    CAS_CBN_ckpts.append(a['ckpts'])
    CAS_CBN_acc.append(a['acc'])

CAS_CBN_ckpts = CAS_CBN_ckpts[0]
CAS_CBN_acc = np.array(CAS_CBN_acc)
CAS_CBN_acc_mean = CAS_CBN_acc.mean(0)
CAS_CBN_acc_std = CAS_CBN_acc.std(0)

CAS_CGN_ckpts = []
CAS_CGN_acc = []
for d in dirs_CGN_bs64:
    a = np.load(os.path.join(d, 'CAS_acc.npz'))
    CAS_CGN_ckpts.append(a['ckpts'])
    CAS_CGN_acc.append(a['acc'])

CAS_CGN_ckpts = CAS_CGN_ckpts[0]
CAS_CGN_acc = np.array(CAS_CGN_acc)
CAS_CGN_acc_mean = CAS_CGN_acc.mean(0)
CAS_CGN_acc_std = CAS_CGN_acc.std(0)

plt.errorbar(CAS_CBN_ckpts, CAS_CBN_acc_mean, yerr=CAS_CBN_acc_std, color='C0', label="CBN")
plt.errorbar(CAS_CGN_ckpts, CAS_CGN_acc_mean, yerr=CAS_CGN_acc_std, color='C1', label="CGN")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("CAS")
plt.savefig('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/CAS.png', bbox_inches='tight', pad_inches=0.25)
plt.close()

#######################################################

fids = {}
iss = {}
for label in labels:
    fids[label] = []
    iss[label] = []
    for class_label in range(10):
        fids[label].append([])
        iss[label].append([])
        iss[label][class_label] = {}
        iss[label][class_label]['ckpts'] = []
        iss[label][class_label]['mean'] = {}
        iss[label][class_label]['std'] = {}
        fids[label][class_label] = {}
        fids[label][class_label]['ckpts'] = []
        fids[label][class_label]['fids'] = {}

# RECORD FID, IS

for i, label in tqdm(enumerate(labels), total=len(labels)):
    # print("\n NORM TYPE", label, "\n")
    # For each class label
    for class_label in tqdm(range(10)):
        # print("\n CLASS LABEL", class_label, "\n")
        # print("dirs:", len(dirs[i]), "\n")
        # For each dir in dirs[i]
        for d in dirs[i]:
            # print("\n")
            a = np.load(os.path.join(d, 'IS_class{:02d}.npz'.format(class_label)))
            b = np.load(os.path.join(d, 'FID_class{:02d}.npz'.format(class_label)))
            # IS
            for j in range(len(a['ckpts'])):
                ckpt = a['ckpts'][j]
                if ckpt not in iss[label][class_label]['ckpts']:
                    # print(ckpt, "new")
                    iss[label][class_label]['ckpts'].append(ckpt)
                    iss[label][class_label]['mean'][ckpt] = [a['mean'][j]]
                    iss[label][class_label]['std'][ckpt] = [a['std'][j]]
                else:
                    # print(ckpt)
                    iss[label][class_label]['mean'][ckpt].append(a['mean'][j])
                    iss[label][class_label]['std'][ckpt].append(a['std'][j])
            # FID
            for j in range(len(b['ckpts'])):
                ckpt = b['ckpts'][j]
                if ckpt not in fids[label][class_label]['ckpts']:
                    # print(ckpt, "new")
                    fids[label][class_label]['ckpts'].append(ckpt)
                    fids[label][class_label]['fids'][ckpt] = [b['fids'][j]]
                else:
                    # print(ckpt)
                    fids[label][class_label]['fids'][ckpt].append(b['fids'][j])

# PLOT IS

plt.figure(figsize=(14, 6))
for class_label in tqdm(range(10)):
    plt.subplot(2, 5, class_label+1)
    # l = max([len(iss[label][class_label]['ckpts']) for label in labels])
    # plt.errorbar(ckpts, [ref_is_mean[class_label]]*l, yerr=[ref_is_std[class_label]]*l, alpha=0.7)
    plt.plot(ckpts, [ref_is_mean[class_label]]*l, color='C2', alpha=0.7, label='CIFAR10')
    for label in labels:
        ckpts = sorted(iss[label][class_label]['ckpts'])
        means = [iss[label][class_label]['mean'][ckpt] for ckpt in ckpts]
        is_m_means = [np.mean(m) for m in means]
        is_m_stds = [np.std(m) for m in means]
        plt.plot(ckpts, is_m_means, alpha=0.7, label=label)
        # plt.errorbar(ckpts, is_m_means, yerr=is_m_stds, alpha=0.7, label=label)
        # plt.gca().set_xticklabels([str(i//1000)+'k' for i in plt.gca().get_xticks().tolist()])
    # Rest
    # if class_label == 9:
    #     plt.legend(loc='lower right')
    plt.legend()
    # plt.yscale("symlog")
    plt.title(CIFAR10_classes[class_label])
    if class_label == 0 or class_label == 5:
        plt.ylabel("Inception Score")
    if class_label > 4:
        plt.xlabel("Epochs")

# Save
# plt.suptitle("Counter accuracy per class")
plt.subplots_adjust(hspace=.3)
plt.subplots_adjust(wspace=.3)
# plt.savefig(os.path.realpath(os.path.join('/scratch/voletivi/sagan', 'CIFAR10_ISs_per_class.png')), bbox_inches='tight', pad_inches=0.25)
plt.savefig(os.path.realpath(os.path.join('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10/', 'CIFAR10_ISs_per_class_bs.png')), bbox_inches='tight', pad_inches=0.25)
plt.clf()
plt.close()


# PLOT IS AVERAGED ACROSS CLASSES
# plt.figure(figsize=(14, 6))
# l = min([len(iss[label][0]['ckpts']) for label in labels])
ckpts = [0, 5, 10, 15, 20, 30, 40, 50]
ckpts_bs4 = [6, 11, 15, 20, 30, 40, 50]
l = len(ckpts)
for k, label in enumerate(labels):
    is_m = []
    if 'bs4' in label:
        cs = ckpts_bs4
    else:
        cs = ckpts
    for class_label in tqdm(range(10)):
        # plt.subplot(2, 5, class_label+1)
        # ckpts = sorted(iss[label][class_label]['ckpts'])[:l]
        if 'bs4' in label:
            is_means = [iss[labels[k-2]][class_label]['mean'][0]]
        else:
            is_means = []
        is_means += [iss[label][class_label]['mean'][ckpt] for ckpt in cs]
        # import pdb; pdb.set_trace()
        is_m.append([np.mean(i) for i in is_means])
    # Mean, std across classes
    is_mean = np.mean(is_m, axis=0)
    is_std = np.std(is_m, axis=0)
    color = 'C'+str(k//2)
    linestyle = ':' if k % 2 == 0 else '--'
    plt.errorbar(ckpts, is_mean, yerr=is_std, alpha=0.7, label=label, c=color, linestyle=linestyle)


l = len(ckpts)
plt.errorbar(ckpts, [np.mean(ref_is_mean)]*l, yerr=[np.mean(ref_is_std)]*l, color='C2', alpha=0.7, label='CIFAR10')
for k, label in enumerate(labels):
    is_m = []
    for class_label in tqdm(range(10)):
        # plt.subplot(2, 5, class_label+1)
        ckpts = sorted(iss[label][class_label]['ckpts'])[:l]
        is_means = []
        is_means += [iss[label][class_label]['mean'][ckpt] for ckpt in ckpts]
        # import pdb; pdb.set_trace()
        is_m.append([np.mean(i) for i in is_means])
    # Mean, std across classes
    is_mean = np.mean(is_m, axis=0)
    is_std = np.std(is_m, axis=0)
    color = 'C'+str(k)
    # linestyle = ':' if k % 2 == 0 else '--'
    plt.errorbar(ckpts, is_mean, yerr=is_std, alpha=0.7, label=label, c=color, linestyle=linestyle)


# plt.gca().set_xticklabels([str(i//1000)+'k' for i in plt.gca().get_xticks().tolist()])
# Rest
plt.legend()
# plt.yscale("symlog")
plt.ylabel('IS - mean & std across classes')
plt.xlabel("Epochs")

# Save# plt.savefig(os.path.realpath(os.path.join('/scratch/voletivi/sagan', 'CIFAR10_FIDs_per_class.png')), bbox_inches='tight', pad_inches=0.25)
plt.savefig(os.path.realpath(os.path.join('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10', 'CIFAR10_ISs_meanAcrossClasses_bs_wthBS4.png')), bbox_inches='tight', pad_inches=0.25)
plt.clf()
plt.close()


# PLOT FID

plt.figure(figsize=(14, 6))
for class_label in tqdm(range(10)):
    plt.subplot(2, 5, class_label+1)
    for label in labels:
        ckpts = sorted(fids[label][class_label]['ckpts'])
        fids_class = [fids[label][class_label]['fids'][ckpt] for ckpt in ckpts]
        fid_means = [np.mean(f) for f in fids_class]
        fid_stds = [np.std(f) for f in fids_class]
        # plt.errorbar(ckpts, fid_means, yerr=fid_stds, alpha=0.7, label=label)
        plt.plot(ckpts, fid_means, alpha=0.7, label=label)
        # plt.gca().set_xticklabels([str(i//1000)+'k' for i in plt.gca().get_xticks().tolist()])
    # Rest
    if class_label == 9:
        plt.legend()
    # plt.yscale("symlog")
    plt.title(CIFAR10_classes[class_label])
    if class_label == 0 or class_label == 5:
        plt.ylabel("FID")
    if class_label > 4:
        plt.xlabel("Epochs")

# Save
# plt.suptitle("Counter accuracy per class")
plt.subplots_adjust(hspace=.3)
plt.subplots_adjust(wspace=.3)
# plt.savefig(os.path.realpath(os.path.join('/scratch/voletivi/sagan', 'CIFAR10_FIDs_per_class.png')), bbox_inches='tight', pad_inches=0.25)
plt.savefig(os.path.realpath(os.path.join('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10', 'CIFAR10_FIDs_per_class_bs.png')), bbox_inches='tight', pad_inches=0.25)
plt.clf()
plt.close()


# PLOT FID AVERAGED ACROSS CLASSES
# plt.figure(figsize=(14, 6))
# l = min([len(fids[label][0]['ckpts']) for label in labels])

ckpts = [0, 5, 10, 15, 20, 30, 40, 50]
ckpts_bs4 = [6, 11, 15, 20, 30, 40, 50]
# l = len(ckpts)
for k, label in enumerate(labels):
    fid_m = []
    for class_label in tqdm(range(10)):
        # plt.subplot(2, 5, class_label+1)
        # ckpts = sorted(fids[label][class_label]['ckpts'])[:l]
        if 'bs4' in label:
            fids_class = [fids[labels[k-2]][class_label]['fids'][0]]
            cs = ckpts_bs4
        else:
            fids_class = []
            cs = ckpts
        fids_class += [fids[label][class_label]['fids'][ckpt] for ckpt in cs]
        fid_m.append([np.mean(f) for f in fids_class])
    # Mean, std across classes
    fids_mean = np.mean(fid_m, axis=0)
    fids_std = np.std(fid_m, axis=0)
    color = 'C'+str(k//2)
    linestyle = ':' if k % 2 == 0 else '--'
    plt.errorbar(ckpts, fids_mean, yerr=fids_std, alpha=0.7, label=label, c=color, linestyle=linestyle)

# plt.gca().set_xticklabels([str(i//1000)+'k' for i in plt.gca().get_xticks().tolist()])
# Rest
plt.legend()
# plt.yscale("symlog")
plt.ylabel('FID - mean & std across classes')
plt.xlabel("Epochs")

# Save
# plt.suptitle("Counter accuracy per class")
# plt.subplots_adjust(hspace=.3)
# plt.subplots_adjust(wspace=.3)
# plt.savefig(os.path.realpath(os.path.join('/scratch/voletivi/sagan', 'CIFAR10_FIDs_per_class.png')), bbox_inches='tight', pad_inches=0.25)
plt.savefig(os.path.realpath(os.path.join('/home/voletiv/EXPERIMENTS/sngan_christiancosgrove_cifar10', 'CIFAR10_FIDs_meanAcrossClasses_bs_withBS4.png')), bbox_inches='tight', pad_inches=0.25)
plt.clf()
plt.close()


l = len(ckpts)
for k, label in enumerate(labels):
    fid_m = []
    for class_label in tqdm(range(10)):
        # plt.subplot(2, 5, class_label+1)
        # ckpts = sorted(fids[label][class_label]['ckpts'])[:l]
        fids_class = []
        fids_class += [fids[label][class_label]['fids'][ckpt] for ckpt in ckpts]
        fid_m.append([np.mean(f) for f in fids_class])
    # Mean, std across classes
    fids_mean = np.mean(fid_m, axis=0)
    fids_std = np.std(fid_m, axis=0)
    color = 'C'+str(k)
    plt.errorbar(ckpts, fids_mean, yerr=fids_std, alpha=0.7, label=label, c=color, linestyle=linestyle)

# plt.gca().set_xticklabels([str(i//1000)+'k' for i in plt.gca().get_xticks().tolist()])
# Rest
plt.legend()
# plt.yscale("symlog")
plt.ylabel('FID - mean & std across classes')
plt.xlabel("Epochs")
