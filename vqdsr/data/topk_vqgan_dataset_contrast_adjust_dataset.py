import numpy as np
import ffmpeg
import random
import torch
import os
import cv2
from os import path as osp
from torch.nn import functional as F
from torch.utils import data as data
import glob

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_mixed_kernels
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment
from basicsr.utils.matlab_functions import imresize
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from vqdsr.data.data_utils import random_crop

from vqgan.model_multiscale_load_top import VQModel

@DATASET_REGISTRY.register() 
class TopkVQGANContrastDataset(data.Dataset):
    """Anime video datasets with VQD"""

    def __init__(self, opt):
        super(TopkVQGANContrastDataset, self).__init__()
        self.opt = opt
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        self.clip_frames = {}

        self.gt_root = opt['dataroot_gt']

        logger = get_root_logger()

        clip_names = os.listdir(self.gt_root)
        for clip_name in clip_names:
            num_frames = len(glob.glob(osp.join(self.gt_root, clip_name, '*.png')))
            self.keys.extend([f'{clip_name}/{i:08d}' for i in range(num_frames)])
            self.clip_frames[clip_name] = num_frames

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False

        self.iso_blur_range = opt.get('iso_blur_range', [0.2, 4])
        self.aniso_blur_range = opt.get('aniso_blur_range', [0.8, 3])
        self.noise_range = opt.get('noise_range', [0, 10])
        self.crf_range = opt.get('crf_range', [18, 35])
        self.ffmpeg_profile_names = opt.get('ffmpeg_profile_names', ['baseline', 'main', 'high'])
        self.ffmpeg_profile_probs = opt.get('ffmpeg_profile_probs', [0.1, 0.2, 0.7])
        self.contrasts = opt.get('contrasts')

        self.scale = opt.get('scale', 4)
        assert self.scale in (2, 4)

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        self.rank, self.world_size = get_dist_info()

         # vqgan config
        self.level_prob=opt['vqgan']['level_prob']
        self.ms_vqgan = VQModel(**opt['vqgan']['params'])
        ckpt_path = opt['vqgan']['load_from']

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.ms_vqgan.load_state_dict(sd['state_dict'], strict=False)
            logger.info(f'load vqgan model path for {self.rank} {self.world_size}: {ckpt_path}\n')
        self.ms_vqgan.eval()

    def get_gt_clip(self, index):
        """
        get the GT(hr) clip with self.num_frame frames
        :param index: the index from __getitem__
        :return: a list of images, with numpy(cv2) format
        """
        key = self.keys[index]  # get clip from this key frame (if possible)
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the "interval" of neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        center_frame_idx = int(frame_name)
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval

        # if the index doesn't satisfy the requirement, resample it
        if (start_frame_idx < 0) or (end_frame_idx >= self.clip_frames[clip_name]):
            center_frame_idx = random.randint(self.num_half_frames * interval,
                                              self.clip_frames[clip_name] - 1 - self.num_half_frames * interval)
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = center_frame_idx + self.num_half_frames * interval

        # determine the neighbor frames
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = osp.join(self.gt_root, clip_name, f'{neighbor:08d}.png')

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # random crop
        img_gts = random_crop(img_gts, self.opt['gt_size']) 
        # augmentation
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot'])

        return img_gts

    def preprocess_vqgan(self, x):
        x = 2.*x - 1.
        return x

    def postprocess_vqgan(self, x):
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
        return x

    @torch.no_grad()
    def vqgan(self, x, k1=1,k2=1,k3=1):
        x = self.ms_vqgan(self.preprocess_vqgan(x), k1,k2,k3)
        x = self.postprocess_vqgan(x)

        return x

    @torch.no_grad()
    def down_sample(self, x):
        x = x.split(1, dim = 0)
        x = [imresize(img[0].cpu(), 1/4).unsqueeze(0) for img in x]
        x = torch.cat(x, dim=0)
        return x

    @torch.no_grad()
    def ajust_contrast(self, img, contrast=0):
        # contrast >= 0
        img = img.astype(np.float32)
        out = img - (img-127.0)*contrast / 255.0  # adjust to lower contract 
        out = np.clip(out, 0, 255)
        return out

    def add_ffmpeg_compression(self, img_lqs, width, height, de_level):
        # ffmpeg
        loglevel = 'error'
        format = 'h264'
        fps = 25
        crf = np.random.uniform(self.crf_range[0], self.crf_range[1])

        try:
            extra_args = dict()
            if format == 'h264':
                vcodec = 'libx264'
                profile = random.choices(self.ffmpeg_profile_names, self.ffmpeg_profile_probs)[0]
                extra_args['profile:v'] = profile

            ffmpeg_img2video = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}',
                             r=fps).filter('fps', fps=fps, round='up').output(
                                 'pipe:', format=format, pix_fmt='yuv420p', crf=crf, vcodec=vcodec,
                                 **extra_args).global_args('-hide_banner').global_args('-loglevel', loglevel).run_async(
                                     pipe_stdin=True, pipe_stdout=True))
            ffmpeg_video2img = (
                ffmpeg.input('pipe:', format=format).output('pipe:', format='rawvideo',
                                                            pix_fmt='rgb24').global_args('-hide_banner').global_args(
                                                                '-loglevel',
                                                                loglevel).run_async(pipe_stdin=True, pipe_stdout=True))

            # read a sequence of images
            for img_lq in img_lqs:
                ffmpeg_img2video.stdin.write(img_lq.astype(np.uint8).tobytes())

            ffmpeg_img2video.stdin.close()
            video_bytes = ffmpeg_img2video.stdout.read()
            ffmpeg_img2video.wait()

            # ffmpeg: video to images
            ffmpeg_video2img.stdin.write(video_bytes)
            ffmpeg_video2img.stdin.close()
            img_lqs_ffmpeg = []
            while True:
                in_bytes = ffmpeg_video2img.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))
                # contrast ajust
                in_frame = self.ajust_contrast(in_frame, de_level)
                
                in_frame = in_frame.astype(np.float32) / 255.
                img_lqs_ffmpeg.append(in_frame)

            ffmpeg_video2img.wait()

            assert len(img_lqs_ffmpeg) == self.num_frame, 'Wrong length'
        except AssertionError as error:
            logger = get_root_logger()
            logger.warn(f'ffmpeg assertion error: {error}')
        except Exception as error:
            logger = get_root_logger()
            logger.warn(f'ffmpeg exception error: {error}')
        else:
            img_lqs = img_lqs_ffmpeg

        return img_lqs
    
    @torch.no_grad()
    def custom_resize(self, x, scale=4):
        h, w = x.shape[2:]
        width = w // scale
        height = h // scale
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        if mode == 'area':
            align_corners = None
        else:
            align_corners = False
        x = F.interpolate(x, size=(height, width), mode=mode, align_corners=align_corners)

        return x

    @torch.no_grad()
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        img_gts = self.get_gt_clip(index)

        # ------------- generate LQ frames --------------#
        # change to CUDA implementation
        img_gts = img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0).to(f'cuda:{self.rank}')
        self.ms_vqgan = self.ms_vqgan.to(f'cuda:{self.rank}')
       
        # add blur
        kernel = random_mixed_kernels(['iso', 'aniso'], [0.7, 0.3], 21, self.iso_blur_range, self.aniso_blur_range)
        with torch.no_grad():
            kernel = torch.FloatTensor(kernel).unsqueeze(0).expand(self.num_frame, 21, 21).to(f'cuda:{self.rank}')
            img_lqs = filter2D(img_gts, kernel)
            # add noise
            img_lqs = random_add_gaussian_noise_pt(
                img_lqs, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=0.5)
            # downsample
            img_lqs = self.custom_resize(img_lqs, self.scale)

            # topk vqgan
            img_lqs = img_lqs.to(f'cuda:{self.rank}')
            de_level = random.random()
            if de_level < self.level_prob[0]:
                k=random.randint(1,15)
            elif de_level < self.level_prob[1]:
                k=random.randint(16,30)
            else:
                k=random.randint(31,50)
            img_lqs = self.vqgan(img_lqs, k3=k)

            height, width = img_lqs.shape[2:]
            # back to numpy since ffmpeg compression operate on cpu
            img_lqs = img_lqs.detach().clamp_(0, 1).permute(0, 2, 3, 1) * 255  # B, H, W, C
            img_lqs = img_lqs.type(torch.uint8).cpu().numpy()[:, :, :, ::-1]
            img_lqs = np.split(img_lqs, self.num_frame, axis=0)
            img_lqs = [img_lq[0] for img_lq in img_lqs]

        # ffmpeg + contrast ajust
        contrast = (k/50.0)*(self.contrasts[1]-self.contrasts[0])+self.contrasts[0]
        img_lqs = self.add_ffmpeg_compression(img_lqs, width, height, contrast)
        # ------------- end --------------#
        img_lqs = img2tensor(img_lqs)
        img_lqs = torch.stack(img_lqs, dim=0)

        return {'lq': img_lqs.cpu(), 'gt': img_gts.cpu()}

    def __len__(self):
        return len(self.keys)
