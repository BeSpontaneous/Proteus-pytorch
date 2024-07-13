# Pre-training on ImageNet-1K


## Installation
Please follow the installation instructions in [DINOv2](https://github.com/facebookresearch/dinov2/tree/main?tab=readme-ov-file#installation) and install timm==0.9.16 as well.

## Dataset
We prepare ImageNet-1K following the instructions in [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md#data-preparation).

## Training
1. Specify the directory of datasets with `data-path` in the training script `run_pretrain.sh`.
2. Use the `teacher-model` and `target_model` parameters to select the appropriate teacher and student models.
3. Specify the model choices with `model` to choose from DINOv2, SynCLR, CLIP.
4. For SynCLR and CLIP training, use the `teacher-path` parameter to indicate the path to the pre-trained teacher model.
5. Simply run the training script as follows:

   ```
   bash run_pretrain.sh
   ```


## Acknowledgment

This part is heavily build upon [DeiT](https://github.com/facebookresearch/deit?tab=readme-ov-file), [DINOv2](https://github.com/facebookresearch/dinov2), [SynCLR](https://github.com/google-research/syn-rep-learn/tree/main/SynCLR). We gratefully thank the authors for their wonderful works.