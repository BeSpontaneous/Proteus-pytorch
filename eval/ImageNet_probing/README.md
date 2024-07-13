# Linear Evaluation on ImageNet-1K


## Installation
Please follow the installation instructions in `pretrain`.

## Dataset
We prepare ImageNet-1K following the instructions in [DINOv2](https://github.com/facebookresearch/dinov2/tree/main?tab=readme-ov-file#data-preparation) (there are additional files needed compared to the pre-training stage).

## Training
1. Specify the config file with `config-file` in the training script `run_probing.sh`.
2. Use the `pretrained-weights` parameter to provide the path to your pre-trained model.
3. Replace `imagenet_path` in the `train-dataset` and `val-dataset` parameters with the directory where your datasets are located.
4. Simply run the training script as follows:

   ```
   bash run_probing.sh
   ```

## Acknowledgment

This part is heavily build upon [DINOv2](https://github.com/facebookresearch/dinov2). We gratefully thank the authors for their wonderful works.