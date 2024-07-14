# Semantic Segmentation and Depth Estimation


## Installation
Please follow the installation instructions in [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation).

## Dataset
Please follow the guidelines in mmsegmentation to prepare [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#ade20k) for Semantic Segmentation and [NYU-Depth V2](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nyu) for Depth Estimation.

## Training
1. Run the training script as follows to modify the pretrained checkpoint format:

   ```
   bash prepare_ckpt.sh
   ```
2. Specify the config file from `configs` in `run.sh`.
3. Simply run the training script as follows:

   ```
   bash run.sh
   ```

## Acknowledgment

This part is heavily build upon [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main) and [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main). We gratefully thank the authors for their wonderful works.