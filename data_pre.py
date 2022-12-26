import argparse
import os
import numpy as np
import torch
import cv2
from multiprocessing import Pool
from tqdm import tqdm
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay



mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)


def extract_crops(video_path, output_path, imsize):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if len(os.listdir(video_path)) == len(os.listdir(output_path)):
        return

    crop_align_func = FasterCropAlignXRay(imsize)
    frames = []
    for f in os.listdir(video_path):
        frame = cv2.imread(os.path.join(video_path, f))
        if frame is None:
            print(frame)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    detect_res, all_lm68 = detect_all(
        file=video_path, frames=frames
    )
    shape = frames[0].shape[:2]

    all_detect_res = []

    assert len(all_lm68) == len(detect_res)

    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)

    detect_res = all_detect_res

    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):

        assert len(detect_res[start:end]) == len(track)

        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]

            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left

            new_box = (box.reshape(2, 2) - top_left).reshape(-1)

            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]

            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx

            frame_boxes[frame_idx] = np.rint(box).astype(np.int32)

    landmarks_list = []
    image_list = []
    for j in range(len(tracks[0])):
        image = data_storage[f"{0}_{j}_img"]
        landmark = data_storage[f"{0}_{j}_ldm"]
        image_list.append(image)
        landmarks_list.append(landmark)

    landmarks, images = crop_align_func(landmarks_list, image_list)

    for image_num, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(
            output_path, '{:04d}.png'.format(image_num)), image)

    print(f'finishing {output_path}')

def extract_videos(output_path, dataset_path, subset, compression, imsize, num_process):
    print(f'Extracting {subset} crops ...')
    videos_path = os.path.join(dataset_path, compression, subset)
    # pool = Pool(num_process)
    for video_file in tqdm(os.listdir(videos_path)):
        extract_crops(
            os.path.join(videos_path, video_file),
            os.path.join(output_path, compression, subset, video_file), imsize
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract face crops of FF++ dataset for FTCN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--output_path', type=str, default='',
                        help='the root path of the output images')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='the root path of the dataset')
    parser.add_argument('--subset', type=str, default='youtube',
                        choices=['youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                        help='which subset of the dataset')
    parser.add_argument('--compression', type=str, default='c0',
                        choices=['c0', 'c23', 'c40'])
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--num_process', type=int, default=1,
                        help='the number of processes')

    args = parser.parse_args()
    extract_videos(**vars(args))