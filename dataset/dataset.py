import bisect
import os
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class ForensicsClips(Dataset):
    """Dataset class for FaceForensics++. Supports returning only a subset of forgery
    methods in dataset"""
    def __init__(
            self,
            mode,
            frames_per_clip,
            fakes=('Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'),
            compression='c0',
            grayscale=False,
            transform=None,
            max_frames_per_video=270,
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        ds_types = ['youtube'] + fakes  # Since we compute AUC, we need to include the Real dataset as well
        for ds_type in ds_types:

            # get list of video names
            videos = []
            video_paths = os.path.join('./dataset/FaceForensics', compression, ds_type)
            print(video_paths)
            with open(os.path.join('./dataset/FaceForensics/split', mode + '.txt')) as f:
                nums = [a.strip() for a in f.readlines()]
            if ds_type == 'youtube':
                for num in nums:
                    i = num.split('_')[0]
                    videos.append(i)
            else:
                for num in nums:
                    videos.append(num)

            self.videos_per_type[ds_type] = len(videos)
            for video in videos:
                path = os.path.join(video_paths, video)
                frame_len = len(os.listdir(path))
                num_clips = 12
                if (frame_len - self.frames_per_clip + 1) < num_clips:
                    continue
                # num_frames = min(len(os.listdir(path)), max_frames_per_video)

                # override fix clip
                # num_clips = num_frames // frames_per_clip

                # random 12 clip per video

                # index = list(range(num_frames))
                # frame_range = [
                #     index[i: i + self.frames_per_clip] for i in range(num_frames) if i + self.frames_per_clip<= num_frames
                # ]
                # num_clips = len(frame_range)

                self.clips_per_video.append(num_clips)
                self.paths.append(path)
        print("videos_per_type")
        print(self.videos_per_type)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        print("clip_length")
        print(clip_lengths.size())
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        print(len(self.cumulative_sizes))
        print(self.cumulative_sizes[-1])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)  # upper bound
        # if video_idx == 0:
        #     clip_idx = idx
        # else:
        #     clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.paths[video_idx]
        frames = sorted(os.listdir(path))
        # random sample clip per epoch
        frame_range = [
            frames[i: i + self.frames_per_clip] for i in range(len(frames)) if i + self.frames_per_clip <= len(frames)
        ]
        # random_clip = random.sample(frame_range, 1)
        random_clip = random.choice(frame_range)

        # start_idx = clip_idx * self.frames_per_clip
        #
        # end_idx = start_idx + self.frames_per_clip

        sample = []
        # for idx in range(start_idx, end_idx, 1):
        #     with Image.open(os.path.join(path, frames[idx])) as pil_img:
        #         if self.grayscale:
        #             pil_img = pil_img.convert("L")
        #         if self.transform is not None:
        #             img = self.transform(pil_img)
        #     sample.append(img)
        for idx in random_clip:
            with Image.open(os.path.join(path, idx)) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                if self.transform is not None:
                    img = self.transform(pil_img)
            sample.append(img)

        return sample, video_idx

    def __getitem__(self, idx):
        sample, video_idx = self.get_clip(idx)

        label = 0 if video_idx < self.videos_per_type['youtube'] else 1  # fake -> 1, real -> 0
        label = torch.from_numpy(np.array(label))
        sample = torch.stack(sample, dim=0)
        sample = sample.permute(1, 0, 2, 3)

        return sample, label, video_idx

