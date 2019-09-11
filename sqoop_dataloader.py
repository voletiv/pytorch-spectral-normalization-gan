import h5py
import io
import json
import os
import numpy as np

from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SqoopDataset(Dataset):

    @property
    def has_answers(self):
        return not (self._all_answers is None)

    def __init__(self, data_root, partition, bug_of_above_vs_below=False, get_scene=False):
        super().__init__()
        self._data_root = data_root
        self._partition = partition
        self._feature_h5 = os.path.join(data_root, f'{partition}_features.h5')
        self._question_h5 = os.path.join(data_root, f'{partition}_questions.h5')
        self.bug_of_above_vs_below = bug_of_above_vs_below
        self.get_scene = get_scene

        # with h5py.File(self._question_h5, 'r') as h5file:
        #     self._all_questions = h5file['questions'][:]
        #     self._all_answers = None
        #     if 'answers' in h5file:
        #         self._all_answers = h5file['answers'][:]
        #         self._all_image_idxs = h5file['image_idxs'][:]

        with h5py.File(self._question_h5, 'r') as f:
            self.len = len(f['questions'])

        if self.get_scene:
            with open(os.path.join(data_root, f'{partition}_scenes.json'), 'r') as f:
                self._scenes = json.load(f)

    def sqoop_collate_fn_batch_read_feats(self, batch):
        if self.get_scene:
            idxs = [b[0] for b in batch]
            scenes = [b[1] for b in batch]
        else:
            idxs = batch

        sorted_idxs, inv_idxs = np.unique(idxs, return_inverse=True)
        with h5py.File(self._question_h5, 'r') as h5file:
            image_idxs = h5file['image_idxs'][sorted_idxs.tolist()][inv_idxs]
            qs = h5file['questions'][sorted_idxs.tolist()][inv_idxs]
            ans = h5file['answers'][sorted_idxs.tolist()][inv_idxs]

        # Images
        im_ids_uniq, inv_indices = np.unique(image_idxs, return_inverse=True)
        with h5py.File(self._feature_h5, 'r') as h5file:
            feats = h5file['features'][im_ids_uniq.tolist()][inv_indices]
        # Make image, choose ONLY green channel
        images = torch.tensor(
            [np.array(Image.open(io.BytesIO(feat))).transpose(
                2, 0, 1)[np.newaxis, 1, :, :] / np.float32(128.01) * 2. - 1.
             for feat in feats]
        )

        # Questions
        # qs === [obj1, relation, obj2]
        # obj1 & obj2 -> -7 because 'A' starts at 7
        # relation -> -43 because "left of"=43, "right of"=44, "above"=45, "below"=46
        # (but actually when "above", it is actually below, and vice versa)
        qs = torch.tensor(qs) - torch.tensor([7, 43, 7])
        # qs = pad_sequence([torch.tensor(q) for q in qs],
        #                   batch_first=True, padding_value=0)

        # Answers
        ans = torch.tensor(ans)

        # Condition: [obj1, rel, obj2]
        # Make only true relations - if answer is no, change relation to the opposite
        cond = qs.clone()
        idx_left_no_to_right = torch.mul(qs[:, 1]==0, ans==0)
        cond[idx_left_no_to_right, 1] = 1 * torch.ones(idx_left_no_to_right.sum(), dtype=cond.dtype)
        idx_right_no_to_left = torch.mul(qs[:, 1]==1, ans==0)
        cond[idx_right_no_to_left, 1] = 0 * torch.ones(idx_right_no_to_left.sum(), dtype=cond.dtype)
        # (and BECAUSE OF BUG in case of above and below, change relation if the answer is yes)
        if self.bug_of_above_vs_below:
            idx_above_yes_to_below = torch.mul(qs[:, 1]==2, ans==1)
            cond[idx_above_yes_to_below, 1] = 3 * torch.ones(idx_above_yes_to_below.sum(), dtype=cond.dtype)
            idx_below_yes_to_above = torch.mul(qs[:, 1]==3, ans==1)
            cond[idx_below_yes_to_above, 1] = 2 * torch.ones(idx_below_yes_to_above.sum(), dtype=cond.dtype)
        else:
            idx_above_yes_to_below = torch.mul(qs[:, 1]==2, ans==0)
            cond[idx_above_yes_to_below, 1] = 3 * torch.ones(idx_above_yes_to_below.sum(), dtype=cond.dtype)
            idx_below_yes_to_above = torch.mul(qs[:, 1]==3, ans==0)
            cond[idx_below_yes_to_above, 1] = 2 * torch.ones(idx_below_yes_to_above.sum(), dtype=cond.dtype)

        # Scene
        if self.get_scene:
            # Scenes - choose only letter and pos
            scenes = torch.tensor(scenes)
            return [images, cond, scenes]
        else:
            return [images, cond]

    def decode_cond(self, cond):
        def decode(s, mode='shape'):
            shapes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
            rels = [' left of ', ' right of ', ' above ', ' below ']
            if mode=='rel':
                return rels[s]
            else:
                return shapes[s]
        return [decode(c[0].item()) + decode(c[1].item(), 'rel') + decode(c[2].item()) for c in cond]

    def str2vocab_idx(self, s):
        ascii_index = ord(s)
        # 0
        if ascii_index == 48:
            return 35
        # Other numbers
        elif ascii_index < 65:
            return ascii_index - 23
        # Alphabets
        else:
            return ascii_index - 65

    def __getitem__(self, index):
        # Get image, q, a from the collate_fn
        if self.get_scene:
            # Scene - [ [shape, pos_x, pos_y], [...], [...], [...], [...] ]
            scene = self._scenes[index]
            scene = [[self.str2vocab_idx(obj['shape'])] + obj['pos'] for obj in scene]
            np.random.shuffle(scene)
            return (index, scene)
        else:
            return index

    def __len__(self):
        return self.len


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    for p in ('val', 'test'):
        dataset = SqoopDataset(data_root='/scratch/michals/sqoop/sqoop-variety_1-repeats_30000',
                               vocab=None, partition=p)
        if dataset.has_answers:
            print(f'{p} has answers')
        else:
            print(f'{p} does not have answers')
    loader = DataLoader(
        dataset, batch_size=8,
        shuffle=False, num_workers=0,
        drop_last=False, pin_memory=True
    )
    print(len(loader.dataset))


# from sqoop.sqoop_dataloader import *
# from torch.utils.data import DataLoader
# ds = SqoopDataset(data_root='/home/voletiv/Datasets/sqoop/sqoop-variety_1-repeats_30000', partition='val')
# dl = DataLoader(ds, batch_size=2000, shuffle=False, num_workers=0, drop_last=False, pin_memory=True,
#                     collate_fn=lambda batch: ds.sqoop_collate_fn_batch_read_feats(batch))
# di = iter(dl); images, conds = next(di)
