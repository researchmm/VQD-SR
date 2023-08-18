# How to Train VQD-SR

The training of VQD-SR contains two stages:
1. Training VQ degradation model on RAL (or other customized LR animation dataset).
3. Training VSR model with VQ degration model on AVC-Train/enhanced AVC-Train (or other customized HR animation dataset).

You can choose to skip any stage by loading the prior models from [google drive](https://drive.google.com/file/d/1MvDG9NfZjnW0kyCyPtokgC3M8lhFgnLv/view?usp=drive_link). 

For example, if you just want to train a VSR model for animation, we recommend you to load the off-the-shelf VQ degradation model.

As described in the paper, all the training is performed on 8 NVIDIA V100 GPUs. You may need to adjust the batchsize according to the CUDA memory of your devices.

## Stage 1: Training VQ Degradation Model
### Dataset Preparation
We use the RAL dataset for the training of VQ degradation model. The RAL dataset is released under request, please refer to [Request for RAL Dataset](README.md#request-for-ral-dataset).

After you download the RAL dataset, put the downloaded data to a root path $dirpath，and modify the [config](taming-transformers/configs) files with $dirpath accordingly.

If you want to use customized LR animation dataset, remember to also generate [dataset config files](taming-transformers/data) containing all the paths of training and test images relative to a root path $dirpath.

### Training
1. Train top-scale VQGAN
   
   Before the training, you should modify the [yaml config file](taming-transformers/configs/top_scale_pretrain.yaml) accordingly. 
   ```bash
   cd taming-transformers
   python main.py -n single_scale --base configs/top_scale_pretrain.yaml -t True --gpus 0,1,2,3,4,5,6,7

   ```
3. Train multi-scale VQGAN

   The GAN model is fine-tuned from the top-scale VQGAN model trained in the previous step. You can also load our pre-trained model [pretrain_top.ckpt](https://drive.google.com/file/d/1MvDG9NfZjnW0kyCyPtokgC3M8lhFgnLv/view?usp=drive_link) 
   ```bash
   cd taming-transformers
   python main.py -n multi_scale --base configs/load_top_mul_scale.yaml -t True --gpus 0,1,2,3,4,5,6,7

   ```

## Stage 2: Training VSR Model
### Dataset Preparation
We use the [AVC-Train](https://github.com/TencentARC/AnimeSR#request-for-avc-dataset)/enhanced AVC-Train with HR-SR strategy dataset for the training of VSR model.

After you download the dataset, put the downloaded data to a root path $dataroot_gt，and modify the [config](options) file with $dataroot_gt accordingly.

In the paper, we utilize a tiny small sized [RealESR](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md) model and bicubic downsampling (MATLAB implementation) to enhance the training HR data. You can also try some other SR models as discussed in the ablation study.

If you want to use customized animation dataset, the data structure should be:
 ```
  ├────$dataroot_gt
        ├────$video1
              ├────xxx.png
              ├────...
              ├────xxx.png
        ├────$video1
              ├────xxx.png
              ├────...
              ├────xxx.png
 ```

### Training
1. Train net model
   ```bash
   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --master_port 1220 --nproc_per_node=8 vqdsr/train.py -opt options/train_vqdsr_net.yml --launcher pytorch [--auto_resume]
   ```
3. Train gan model
   
   Before training, remember to modify the configuration of VQ degradation model $vqgan in the [yaml config](options/train_vqdsr_gan.yml)
    ```bash
   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --master_port 1220 --nproc_per_node=8 vqdsr/train.py -opt options/train_vqdsr_gan.yml --launcher pytorch [--auto_resume]
   ```
## Evaluation
We follow the [prior work](https://github.com/TencentARC/AnimeSR/blob/main/scripts/metrics/README.md) for the evaluation of [MANIQA](https://github.com/IIGROUP/MANIQA).

However, we parallelly evaluate the resulting frames with multiple randomly selected cropping sets considering the randomness. We provided the cropping coordinates we used under ['crop_meta'](scripts/metrics/crop_meta) for you to reproduce the results in our paper.

You can also generate random cropping sets by the [script](scripts/metrics/generate_random_crop_region.py) we prvided.

Multiple parallel evaluations:
  ```bash
   cd scripts/metrics/MANIQA
   python inference_fix_multi_crop.py --model_path ckpt_ensemble --input_dir $path_of_result_folder --output_dir $output/ --crop_meta ../crop_meta
   ```

## The Pre-Trained Checkpoints
You can download the checkpoints of all models in [google drive](https://drive.google.com/file/d/1MvDG9NfZjnW0kyCyPtokgC3M8lhFgnLv/view?usp=drive_link).
