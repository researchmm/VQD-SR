import cv2
import os
import glob
from os import path as osp
import argparse
from vqdsr.utils.video_util import frames2video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input frames root path')
    parser.add_argument('-o', '--output', type=str, help='output videos root path')
    parser.add_argument('--fps', type=int, default=24, help='fps of out videos')
    args = parser.parse_args()
    in_root = args.input
    out_root = args.output
    fps = args.fps
    os.makedirs(out_root, exist_ok=True)

    videos_name = sorted(os.listdir(in_root))
    video_output = osp.join(out_root, 'videos')
    os.makedirs(video_output, exist_ok=True)
    for video_name in videos_name:
        out_path = osp.join(video_output, f'{video_name}.mp4')
        frames2video(
            osp.join(in_root, video_name), out_path, fps=fps, suffix='png')

if __name__ == '__main__':
    main()

