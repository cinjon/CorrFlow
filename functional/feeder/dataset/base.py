import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.img import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class BaseDataset(Dataset):
    """The parent dataset class.

    Args:
      train: Whether this is training. Obviates trainval.
      trainval: Whether this is trainval.
      num_samples: The number of samples to use from within the frame range.
    """

    # TODO: Add in the masking info when it's done computing.

    def __init__(self, hparams, shuffle=False):
        self.img_size = hparams.img_size
        self.crop_size = hparams.crop_size
        self.crop_size2 = hparams.crop_size2
        self.video_len = hparams.video_len
        self.pred_distance = hparams.pred_distance  # how many frames away
        self.offset = hparams.offset  # offset x, y parameters
        self.grid_size = hparams.grid_size
        self.frame_gap = hparams.frame_gap
        self.use_masks = hparams.use_masks

        self.folder_paths = []
        self.frame_nums = []

        print('Filtering file...')
        self._total_frames = 0
        with open(self.file_list, 'r') as f:
            for line in f:
                folder_path, frame_num = line.split()[:2]
                frame_num = int(frame_num)
                self.folder_paths.append(folder_path)
                self.frame_nums.append(frame_num)
                self._total_frames += frame_num
        size = len(self.folder_paths)
        print(
            f"Size of dataset is {size}. Total frames is {self._total_frames}")

        # if shuffle:
        #     order = list(range(len(self.folder_paths)))
        #     random.shuffle(order)
        #     self.folder_paths = [self.folder_paths[i] for i in order]
        #     self.frame_nums = [self.frame_nums[i] for i in order]
        #     del order

        self.geometric_transformation = GeometricTransformation(
            'affine',
            out_h=self.crop_size2,  # defaults to 80
            out_w=self.crop_size2,
            use_cuda=False)

    @property
    def total_frames(self):
        return self._total_frames

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, index):
        folder_path = self.folder_paths[index]
        frame_num = self.frame_nums[index]

        # Instantiate the imgs, imgs_target, patch_target, and future_imgs.
        imgs = torch.Tensor(self.video_len, 3, self.crop_size, self.crop_size)
        imgs_target = torch.Tensor(2, 3, self.crop_size, self.crop_size)
        patch_target = torch.Tensor(2, 3, self.crop_size, self.crop_size)
        future_imgs = torch.Tensor(2, 3, self.crop_size, self.crop_size)

        # Should we flip all of the images?
        to_flip = True if random.random() <= 0.5 else False

        # TODO:
        # What is video_len? I am guessing that pred_distance is the cycle length to consider.
        # And that frame_gap is how many in between to consider. This is default to 2.
        # Then current_len would be the number of frames in total that this looks at.
        current_len = (self.video_len + self.pred_distance) * self.frame_gap
        # future_index is ~ start_frame + current_len - 1, i.e. the last frame we see
        # before cycling back.
        start_frame, future_index, frame_gap = self._get_index_values(
            frame_num, current_len, self.frame_gap)

        # Load the transformed images. Note that we select the crop_offsets and then reuse them.
        crop_offset_x = -1
        crop_offset_y = -1
        for i in range(self.video_len):
            img = self._load_img(start_frame, i, folder_path, frame_gap)
            if self.use_masks:
                mask = self._load_mask(start_frame, i, folder_path, frame_gap)

            img, crop_offset_x, crop_offset_y = self._transform(img,
                                                                crop_offset_x,
                                                                crop_offset_y,
                                                                to_flip=to_flip)
            imgs[i] = img.clone()

        # Do the same for future_imgs. Use the same crop_offset_x and crop_offset_y.
        # Note that we'll take two here but in the end use only the first. The second
        # is used to generate the flow fields.
        for i in range(2):
            img = self._load_future(future_index,
                                    i,
                                    folder_path,
                                    frame_gap=frame_gap,
                                    frame_num=frame_num)
            img, crop_offset_x, crop_offset_y = self._transform(img,
                                                                crop_offset_x,
                                                                crop_offset_y,
                                                                to_flip=to_flip)
            future_imgs[i] = img
            imgs_target[i] = future_imgs[i].clone()

        # Compute the diff between the two future_imgs. This is used further down to select the patch.
        flow_cmb = future_imgs[0] - future_imgs[1]
        flow_cmb = img_to_np(flow_cmb)
        flow_cmb = flow_cmb.astype(np.float)
        flow_cmb = np.abs(flow_cmb)

        # A typical value is crop_size = 240, grid_size = 9. So box_edge = 26.
        side_edge = self.crop_size
        box_edge = int(side_edge / self.grid_size)

        lblxset = []
        lblyset = []
        scores = []

        # Take the (grid_size-2)^2 overlapping patches of size (box_edge*3)^2.
        # For each of them, their score is how much they change according to flow_cmb.
        for i in range(self.grid_size - 2):
            for j in range(self.grid_size - 2):
                offset_x1 = i * box_edge
                offset_y1 = j * box_edge
                lblxset.append(i)
                lblyset.append(j)

                offset_y1_end = offset_y1 + box_edge * 3
                offset_x1_end = offset_x1 + box_edge * 3
                tpatch = flow_cmb[offset_y1:offset_y1_end, offset_x1:
                                  offset_x1_end].copy()
                tsum = tpatch.sum()
                scores.append(tsum)

        # Sort the scores and choose a random patch in the top 10.
        scores = np.array(scores)
        ids = np.argsort(scores)[-10:]
        lbl = ids[random.randint(0, 9)]
        lbl_x = lblxset[lbl]
        lbl_y = lblyset[lbl]

        # In case this is a flip, lbl_x needs to be flipped.
        if self.is_train and to_flip:
            lbl_x = self.grid_size - 3 - lbl_x

        # Assuming grid_size=9, then we had 7^2 = 49 choices from we picked
        # one (x, y) pair, e.g. (3, 5). We now convert that to
        # lbl = 3*7 + 5 = 26. This is the return label we are expecting.
        lbl = lbl_x * (self.grid_size - 2) + lbl_y

        # TODO: What is this hardcoded 6 doing here? Should it be (self.grid_size - 1)?
        # Ok, so in the aforementioned example, we now have xloc = 1/2, yloc = 5/6.
        xloc = lbl_x / 6.
        yloc = lbl_y / 6.

        # TODO: What does this theta_aff do?
        theta_aff = np.random.rand(6)
        scale = 2. / 3.
        # randnum is [(rand() - 0.5) / 6, (rand() - 0.5) / 6]
        randnum = (np.random.rand(2) - 0.5) / 6.
        # e.g. xloc is (rand() + 2.5)/6, yloc is (rand() + 4.5)/6, both clipped to 0,1.
        xloc = np.clip(xloc + randnum[0], 0., 1.)
        yloc = np.clip(yloc + randnum[1], 0., 1.)

        alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * (1. / 4.)

        # Make a transformation matrix where the left 2x2 square is R(pi/2 * (rand - 0.5))/3
        # and the right 2x1 column is scale*[(2 * xloc - 1.); (2*yloc - 1.)].
        theta_aff[2] = (xloc * 2. - 1.) * scale
        theta_aff[5] = (yloc * 2. - 1.) * scale
        theta_aff[0] = 1. / 3. * np.cos(alpha)
        theta_aff[1] = 1. / 3. * (-np.sin(alpha))
        theta_aff[3] = 1. / 3. * np.sin(alpha)
        theta_aff[4] = 1. / 3. * np.cos(alpha)
        theta = torch.Tensor(theta_aff.astype(np.float32))
        theta = theta.view(1, 2, 3)

        # Repeat the transformation for each of the imgs_target elements.
        theta_batch = theta.repeat(2, 1, 1)

        # Affine transformation with out_h = out_w = self.crop_size2 (default 80).
        patch_target = self.geometric_transformation(imgs_target, theta_batch)
        theta = theta.view(2, 3)

        # Now limit the target and patch to just the first one.
        imgs_target = imgs_target[:1]
        patch_target = patch_target[:1]

        meta = {
            'folder_path': folder_path,
            'start_frame': start_frame,
            'future_index': future_index,
            'frame_gap': float(frame_gap),
            'crop_offset_x': crop_offset_x,
            'crop_offset_y': crop_offset_y,
            'dataset': self._name
        }

        return imgs, imgs_target, patch_target, theta, meta

    def _transform(self, img, crop_offset_x=-1, crop_offset_y=-1,
                   to_flip=False):
        """Transform the image.

        1. Resize to have the same aspect ratio but with shorter side length self.img_size.
        2. Crop the image at the offset given by crop_offset_x, crop_offset_y.
        3. If crop_offset_x == -1, then we haven't chosen it yet, so do so randomly.
        4. Flip if that is set.
        5. Color normalize according to the global mean and std.

        Args:
          crop_offset_x:
          crop_offset_y:
          to_flip: Whether we left-right flip the image.

        Returns:
          img: Transformed image in torch.Tensor form
        """
        height, width = img.size(1), img.size(2)
        new_height, new_width = height, width
        if height <= width:
            ratio = float(width) / float(height)
            img = resize(img, int(self.img_size * ratio), self.img_size)
        else:
            ratio = float(height) / float(width)
            img = resize(img, self.img_size, int(self.img_size * ratio))

        if crop_offset_x == -1:
            crop_offset_x = random.randint(0, img.size(2) - self.crop_size - 1)
            crop_offset_y = random.randint(0, img.size(1) - self.crop_size - 1)

        img = crop_img(img, crop_offset_x, crop_offset_y, self.crop_size)
        assert (img.size(1) == self.crop_size)
        assert (img.size(2) == self.crop_size)

        if self.is_train and to_flip:
            img = torch.from_numpy(fliplr(img.numpy())).float()

        img = color_normalize(img, mean, std)
        return img, crop_offset_x, crop_offset_y
