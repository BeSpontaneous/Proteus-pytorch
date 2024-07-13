# Linear Evaluation on ImageNet-1K


## Installation
Please follow the installation instructions in `pretrain`.

## Dataset
We prepare ImageNet-1K following the instructions in [DINOv2](https://github.com/facebookresearch/dinov2/tree/main?tab=readme-ov-file#data-preparation) (there are additional files needed compared to the pre-training stage).

## Training
1. Specify the config file with `config-file` in the training script `run_probing.sh`.
2. Specify the path of pre-trained model with `pretrained-weights`.
3. Specify the directory of datasets to repalce `imagenet_path` in `train-dataset` and `val-dataset`.
4. Simply run the training script as follows:

   ```
   bash run_probing.sh
   ```

## Acknowledgment

This part is heavily build upon [DINOv2](https://github.com/facebookresearch/dinov2). We gratefully thank the authors for their wonderful works.