import argparse
import os.path
import torch

def get_base_argument_parser():
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default=None,
        help='Model names: VQD_SR')
    parser.add_argument(
        '-s',
        '--outscale',
        type=int,
        default=4,
        help='The netscale is x4, but you can achieve arbitrary output scale (e.g., x2) with the argument outscale'
        'The program will further perform cheap resize operation after the VQD_SR output. '
        'This is useful when you want to save disk space or avoid too large-resolution output')
    parser.add_argument(
        '--expname', type=str, default='vqdsr', help='A unique name to identify your current inference')
    parser.add_argument(
        '--netscale',
        type=int,
        default=4,
        help='the released models are all x4 models, only change this if you train a x2 or x1 model by yourself')
    parser.add_argument(
        '--mod_scale',
        type=int,
        default=4,
        help='the scale used for mod crop, since VQD_SR use a multi-scale arch, so the edge should be divisible by 4')
    parser.add_argument('--fps', type=int, default=None, help='fps of the sr videos')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')

    return parser


