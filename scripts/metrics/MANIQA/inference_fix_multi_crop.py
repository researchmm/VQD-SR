import argparse
import os
import random
import torch
from pipal_data import NTIRE2022
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import Normalize, ToTensor, crop_image

# CUDA_VISIBLE_DEVICES='' python inference_fix_multi_crop.py --model_path ckpt_ensemble --input_dir ../../../results/ --output_dir output/ --crop_meta ../crop_meta

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script of RealBasicVSR')
    parser.add_argument('--model_path', help='checkpoint file', required=True)
    parser.add_argument('--input_dir', help='directory of the input video', required=True)
    parser.add_argument('--crop_meta', help='directory of the crop meta info', required=True)
    parser.add_argument(
        '--output_dir',
        help='directory of the output results',
        default='output/ensemble_attentionIQA2_finetune_e2/VQDSR')
    args = parser.parse_args()

    return args


def test_single_time(args, crop_region):
    # configuration
    batch_size = 10
    num_workers = 0
    average_iters = 20
    crop_size = 224

    model = torch.load(args.model_path)

    # map to cuda, if available
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True
    model.eval()
    total_avg_score = []
    subfolder_namelist = []
    for subfolder_name in sorted(os.listdir(args.input_dir)):
        avg_score = 0.0
        subfolder_root = os.path.join(args.input_dir, subfolder_name)
        subfolder_meta = os.path.join(crop_region, subfolder_name+'.txt')

        with open(subfolder_meta, 'r') as f:
            posi_infos=f.readlines()

        if os.path.isdir(subfolder_root) and subfolder_name != 'assemble-folder':
            # data load
            val_dataset = NTIRE2022(
                ref_path=subfolder_root,
                dis_path=subfolder_root,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
            val_loader = DataLoader(
                dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

            name_list, pred_list = [], []


            for data in tqdm(val_loader):
                pred = 0

                for posi_info in posi_infos:
                    posi = posi_info.strip().split(',')
                    top, left = int(posi[0]), int(posi[1])
                    if cuda_flag:
                        x_d = data['d_img_org'].cuda()
                
                    img = crop_image(top, left, crop_size, img=x_d)
                    with torch.no_grad():
                        pred += model(img)
                pred /= len(posi_infos)
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)

            for i in range(len(name_list)):
                avg_score += float(pred_list[i][0])

            avg_score /= len(name_list)
            subfolder_namelist.append(subfolder_name)
            total_avg_score.append(avg_score)

    with open(os.path.join(args.output_dir, os.path.basename(crop_region)+'_average.txt'), 'w') as f:
        for idx, averge_score in enumerate(total_avg_score):
            string = f'Folder {subfolder_namelist[idx]}, Average Score: {averge_score:.6f}\n'
            f.write(string)
            print(f'Folder {subfolder_namelist[idx]}, Average Score: {averge_score:.6f}')

        print(f'Average Score of {len(subfolder_namelist)} Folders: {sum(total_avg_score) / len(total_avg_score):.6f}')
        string = f'Average Score of {len(subfolder_namelist)} Folders: {sum(total_avg_score) / len(total_avg_score):.6f}'  # noqa E501
        f.write(string)
        f.close()
    
    return sum(total_avg_score) / len(total_avg_score)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'final.txt'), 'w') as f:
        crop_regions = ["crop_0", "crop_1", "crop_2",] # crop region set
        
        scores = []
        for crop_region in crop_regions:
            crop_dir = os.path.join(args.crop_meta, crop_region)
            ave = test_single_time(args, crop_dir)
            scores.append(ave)
            text = f'{crop_region}: {ave:.6f}\n'
            f.write(text)

        ave_score=sum(scores) / len(scores)
        text = f'averge score: {ave_score:.6f}\n'
        f.write(text)
