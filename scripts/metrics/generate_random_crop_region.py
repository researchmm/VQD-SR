import os
import random

img_size_info = "./reallq_avc_size.txt"  # the original size of each videos in test dataset
crop_size=224
average_iters = 20  # num of crops for each video
num_set = 3  

def generate_one_set(cor_record):
    with open(os.path.join(img_size_info), 'r') as f:
        size_info=f.readlines()

    for line in size_info:
        line = line.split(',')
        subfolder_name, h, w = line[0], int(line[1]), int(line[2])

        with open(os.path.join(cor_record,subfolder_name+'.txt'), 'w') as f:
            for i in range(average_iters):
                top = random.randint(0, h - crop_size)
                left = random.randint(0, w - crop_size)
                string = f'{top},{left}\n'
                f.write(string)

if __name__=='__main__':
    for i in range(num_set):
        cor_record = './crop_meta/crop_'+str(i)
        os.makedirs(cor_record, exist_ok=True)

        generate_one_set(cor_record)

