"""Example command: python benchmark.py --resume /checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.pth --savepath /checkpoint/cinjon/spaceofmotion/supercons/corrflow.kineticsmodel.gymnastics --workers 8 --dataset jun1full-gymnastics --use_max_mask --datasplit trainval --limit 10 --skip_frames 3"""
import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np

import functional.feeder.dataset.DavisLoader as DL
import functional.feeder.dataset.gymnastics as gymnastics
from functional.utils.f_boundary import db_eval_boundary
from functional.utils.jaccard import db_eval_iou
from models.corrflow import CorrFlow
from functional.utils.io import imwrite_indexed, write_img

import logger


def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    if args.dataset == 'davis':
        data = DL.dataloader(args.datapath)
        catnames = DL.catnames
        data_loader = torch.utils.data.DataLoader(
            DL.myImageFloder(data[0], data[1], False),
            batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False
        )
    elif 'gymnastics' in args.dataset:
        is_train = args.datasplit == 'train'
        is_trainval = args.datasplit == 'trainval'
        dataset = gymnastics.get_dataset(args.dataset, args, is_train=is_train, is_trainval=is_trainval)
        catnames = dataset.catnames
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers,
            drop_last=False
        )

    model = CorrFlow(args)
    model = nn.DataParallel(model).cuda()

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    test(data_loader, model, log, catnames, args)

    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def test(dataloader, model, log, catnames, args):
    model.eval()

    Fs = AverageMeter()
    Js = AverageMeter()

    n_b = len(dataloader)

    log.info("Start testing.")
    for b_i, (images_orig, images_rgb, annotations, frames_used, img_path, frame_num) in enumerate(dataloader):
        print('On index : ', b_i, img_path, frame_num)
        images_orig = [img.cpu().numpy()[0] for img in images_orig]
        # print(images_orig[0].shape, images_orig[0].dtype, images_orig[0].max(), images_orig[0].min())
        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]
        frames_used = [int(i) for i in frames_used]

        N = len(images_rgb)

        for i in range(N-1):
            rgb_0 = images_rgb[i]
            rgb_1 = images_rgb[i+1]
            if i == 0:
                anno_0 = annotations[i]
            else:
                anno_0 = output
            anno_1 = annotations[i+1]

            _, _, h, w = anno_0.size()

            with torch.no_grad():
                output = model(rgb_0, anno_0, rgb_1)
                output = F.interpolate(output, (h,w), mode='bilinear')
                output = torch.argmax(output, 1, keepdim=True).float()

            max_class = anno_1.max()
            js, fs = [], []

            for classid in range(1, max_class + 1):
                obj_true = (anno_1 == classid).cpu().numpy()[0, 0]
                obj_pred = (output == classid).cpu().numpy()[0, 0]

                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)

                fs.append(f); js.append(j)

            ###
            _name = ['benchmark']
            if args.datasplit:
                _name.append(args.datasplit)
            if args.offset is not None:
                _name.append('off%d' % args.offset)
            if args.num_frames is not None:
                _name.append('nf%d' % args.num_frames)
            if args.skip_frames is not None:
                _name.append('sf%d' % args.skip_frames)
            if args.use_mask_inconsistencies:
                _name.append('umi')
            if args.use_max_mask:
                _name.append('umm')
            folder = os.path.join(args.savepath, '-'.join(_name))

            if not os.path.exists(folder): os.mkdir(folder)

            try:
                output_folder = os.path.join(folder, catnames[b_i].strip())
            except Exception as e:
                print(folder, b_i, DL.catnames)
                raise

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad = ((0,0),(3,3)) if anno_0.size(3) < 1152 else ((0,0), (0,0))
            frame_index = frames_used[i]
            
            output_file = os.path.join(output_folder, '%s.anno.orig.png' % str(frame_index).zfill(5))
            out_img = annotations[i][0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img )

            output_file = os.path.join(output_folder, '%s.anno.model.png' % str(frame_index).zfill(5))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img)

            output_file = os.path.join(output_folder, '%s.rgb.png' % str(frame_index).zfill(5))
            write_img(output_file, images_orig[i])
            
            f = np.mean(fs); j = np.mean(js)
            Fs.update(f); Js.update(j)

        info = '\t'.join(['Js: ({:.3f}). Fs: ({:.3f}).'
                          .format(Js.avg, Fs.avg)])

        log.info('[{}/{}] {}'.format( b_i, n_b, info ))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CorrFlow')

    # Data options
    parser.add_argument('--datapath', help='Data path for Davis')
    parser.add_argument('--savepath', type=str, default='results/test',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    parser.add_argument('--dataset', type=str, default='davis',
                        help='which dataset to use')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--offset', type=int, default=0,
                        help='the offset in each video that we use')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='the number of frames to run the algorithm for')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='the number of frames to skip in between each frame.')
    parser.add_argument('--use_mask_inconsistencies', default=False, action='store_true',
                        help='whether to use the mask inconsistencies in gymnastics.py')
    parser.add_argument('--use_max_mask', default=False, action='store_true',
                        help='whether to use the max mask count')
    parser.add_argument('--datasplit', type=str, default=None,
                        help='which datasplit to use in gymnastics - train, trainval, test')
    parser.add_argument('--limit', type=int, default=None,
                        help='limit the number of folders to run')
    parser.add_argument('--gymnastics_dataset_location', type=str,
                        default='/checkpoint/cinjon/spaceofmotion/jun-01-2019',
                        help='location for gymnastics dataset')

    args = parser.parse_args()

    main()
