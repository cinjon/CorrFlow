from collections import defaultdict
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


FPS = 25.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def r_loader(path):
    # This should output RGB.
    img = np.load(path) / 255.
    return img.astype(np.float32)
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image

def r_prep(image, M):
    h,w = image.shape[0], image.shape[1]
    if w%M != 0: image = image[:,:-(w%M)]
    if h%M != 0: image = image[:-(h%M),]
    return transforms.ToTensor()(image)

def a_loader(mask_path):
    try:
        masks = np.load(mask_path)
        ret = np.zeros(masks[0].shape).astype(np.uint8)
        for i in range(masks.shape[0]):
            ret[np.where(masks[i] == 1)] = i+1
        return ret
    except Exception as e:
        print('a_loader exception in %s --> %s' % (mask_path, e))
        return None
    
def a_prep(image, M):
    h,w = image.shape[0], image.shape[1]
    if w % M != 0: image = image[:,:-(w%M)]
    if h % M != 0: image = image[:-(h%M),:]
    image = np.expand_dims(image, 0)
    return torch.Tensor(image).contiguous().long()



def get_dataset(dataset, hparams, is_train, is_trainval):
    assert (dataset in [
        'jun1full-gymnastics', 'jun1sub6-gymnastics', 'jun1sub25-gymnastics',
        'jun1sub50-gymnastics', 'jun1sub75-gymnastics'
    ])
    if dataset == 'jun1full-gymnastics':
        dataset = _GymnasticsJun01FullDataset
    elif dataset == 'jun1sub6-gymnastics':
        dataset = _GymnasticsJun01Subset6Dataset
    elif dataset == 'jun1sub25-gymnastics':
        dataset = _GymnasticsJun01Subset25Dataset
    elif dataset == 'jun1sub50-gymnastics':
        dataset = _GymnasticsJun01Subset50Dataset
    elif dataset == 'jun1sub75-gymnastics':
        dataset = _GymnasticsJun01Subset75Dataset

    info_directory = os.path.join(hparams.gymnastics_dataset_location,
                                  dataset.location_suffix)
    return dataset(hparams,
                   is_train=is_train,
                   is_trainval=is_trainval,
                   info_directory=info_directory)


class _SomDataset(Dataset):
    """The parent dataset class.

    Args:
      train: Whether this is training. Obviates trainval.
      trainval: Whether this is trainval.
      num_samples: The number of samples to use from within the frame range.
    """

    def __init__(
            self,
            hparams,
            is_train=True,
            is_trainval=False,
            info_directory=None,
    ):
        self.is_train = is_train
        self.is_trainval = is_trainval
        self.offset = hparams.offset
        self.num_frames = hparams.num_frames
        self.skip_frames = hparams.skip_frames
        self.use_mask_inconsistencies = hparams.use_mask_inconsistencies
        self.use_max_mask = hparams.use_max_mask
        self.gym_loc = info_directory
        
        self.is_train = is_train
        self.is_trainval = is_trainval

        if self.is_train:
            self.file_list = os.path.join(
                self.gym_loc, 'gym_train.txt')
        elif self.is_trainval:
            self.file_list = os.path.join(
                self.gym_loc, 'gym_trainval.txt')
        else:
            self.file_list = os.path.join(
                self.gym_loc, 'gym_test.txt')

        # These are fucking up... not sure why just yet.
        bad_list = ['fulfill-goals-go']
        self.img_paths = []
        self.frame_nums = []
        self.mask_paths = []
        self.catnames = []
        with open(self.file_list, 'r') as f:
            for line in f:
                jpg_file, frame_num = line.split()
                if any([k in jpg_file for k in bad_list]):
                    continue
                self.img_paths.append(jpg_file)
                self.frame_nums.append(int(frame_num))
                self.mask_paths.append(jpg_file.replace('frames', 'masks'))
                self.catnames.append(jpg_file.strip().split('/')[-1])

        index_shuf = list(range(len(self.catnames)))
        random.shuffle(index_shuf)
        if hparams.limit:
            index_shuf = index_shuf[:hparams.limit]
        self.img_paths = [self.img_paths[i] for i in index_shuf]
        self.frame_nums = [self.frame_nums[i] for i in index_shuf]
        self.mask_paths = [self.mask_paths[i] for i in index_shuf]
        self.catnames = [self.catnames[i] for i in index_shuf]

        # When we use the following, we build a list of start indices that are
        # s.t. there is a mask in the first one that is not in the second. We want
        # to propagate that mask.
        if self.use_mask_inconsistencies:
            cat_start_indices = defaultdict(list)
            cat_end_indices = defaultdict(list)
            
            for index in range(len(index_shuf)):
                frame_num = self.frame_nums[index]
                mask_path = self.mask_paths[index]
                end_frame = frame_num
                if self.num_frames is not None:
                    end_frame = min(end_frame, self.offset + self.num_frames)
                    
                prev_mask = None
                mask_max = 0
                for i in range(self.offset, end_frame, self.skip_frames):
                    curr_mask = np.load(os.path.join(mask_path, '{:.4f}.npy'.format(float(i)/FPS)))
                    if prev_mask is None:
                        prev_mask = curr_mask
                        continue

                    prev_num_masks = prev_mask.shape[0]
                    curr_num_masks = curr_mask.shape[0]
                    if prev_num_masks > curr_num_masks and prev_num_masks > mask_max:
                        mask_max = prev_num_masks
                        cat_start_indices[index].append(i-1)
                    elif mask_max > 0 and prev_num_masks < curr_num_masks:
                        mask_max = 0
                        cat_end_indices[index].append(i-1)

                    prev_mask = curr_mask

                if index not in cat_start_indices:
                    print('This category is not represented: ', self.catnames[index])
                    cat_start_indices[index].append(self.offset)
                    cat_end_indices[index].append(end_frame)

                if len(cat_start_indices[index]) > len(cat_end_indices[index]):
                    cat_end_indices[index].append(i-1)

                indices_to_keep = [i for i in range(len(cat_start_indices[index])) \
                                   if cat_end_indices[index][i] - cat_start_indices[index][i] > 3*self.skip_frames]
                cat_start_indices[index] = [cat_start_indices[index][i]
                                            for i in indices_to_keep]
                cat_end_indices[index] = [cat_end_indices[index][i]
                                          for i in indices_to_keep]
                
                print(self.catnames[index])
                print(cat_start_indices[index])
                print(cat_end_indices[index])
                print('\n')
                    
            self.cat_start_indices = cat_start_indices
            self.cat_end_indices = cat_end_indices
        elif self.use_max_mask:
            cat_start_indices = defaultdict(list)
            cat_end_indices = defaultdict(list)

            for index in range(len(index_shuf)):
                frame_num = self.frame_nums[index]
                mask_path = self.mask_paths[index]
                end_frame = frame_num
                if self.num_frames is not None:
                    end_frame = min(end_frame, self.offset + self.num_frames)
            
                masks = [np.load(os.path.join(mask_path, '{:.4f}.npy'.format(float(i)/FPS)))
                         for i in range(self.offset, end_frame)]
                mask_counts = [mask.shape[0] for mask in masks]
                argmax_count = np.argmax(mask_counts)
                cat_start_indices[index].append(argmax_count + self.offset)
                cat_end_indices[index].append(end_frame)

                print(self.catnames[index])
                print(cat_start_indices[index])
                print(cat_end_indices[index])
                print('\n')
                    
            self.cat_start_indices = cat_start_indices
            self.cat_end_indices = cat_end_indices
                    

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        frame_num = self.frame_nums[index]

        if self.use_mask_inconsistencies:
            frame_index = random.choice(list(range(len(self.cat_start_indices[index]))))
            start_frame = self.cat_start_indices[index][frame_index]
            end_frame = self.cat_end_indices[index][frame_index]            
        else:
            start_frame = self.offset
            end_frame = frame_num
            if self.num_frames is not None:
                end_frame = min(frame_num, start_frame + self.num_frames)

        print('getitem path: ' + mask_path)
        annotations = [(i, a_loader(os.path.join(mask_path, '{:.4f}.npy'.format(float(i)/FPS))))
                       for i in range(start_frame, end_frame, self.skip_frames)]
        annotations = [(i, a_prep(anno, M=8)) for i, anno in annotations if anno is not None]

        images_orig = [r_loader(os.path.join(img_path, '{:.4f}.npy'.format(float(i)/FPS)))
                      for i, _ in annotations]
        images_rgb = [r_prep(img, M=8) for img in images_orig]

        frames_used = [i for i, _ in annotations]
        annotations = [anno for _, anno in annotations]
        return images_orig, images_rgb, annotations, frames_used, img_path, frame_num

    def __len__(self):
        return len(self.img_paths)


class _GymnasticsJun01FullDataset(_SomDataset):
    location_suffix = 'full-data'
    _name = 'gymnastics-full'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _GymnasticsJun01Subset6Dataset(_SomDataset):
    location_suffix = 'subset6-data'
    _name = 'gymnastics-sub6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_fraction = 1.


class _GymnasticsJun01Subset25Dataset(_SomDataset):
    location_suffix = 'subset25-data'
    _name = 'gymnastics-sub25'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_fraction = .5


class _GymnasticsJun01Subset50Dataset(_SomDataset):
    location_suffix = 'subset50-data'
    _name = 'gymnastics-sub50'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_fraction = .2


class _GymnasticsJun01Subset75Dataset(_SomDataset):
    location_suffix = 'subset75-data'
    _name = 'gymnastics-sub75'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_fraction = .15


