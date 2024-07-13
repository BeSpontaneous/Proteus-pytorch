# Fine-grained Classification on 12 datasets


## Installation
Please follow the installation instructions in `pretrain`.

## Dataset
There are 12 fine-grained classification datasets in total: `aircraft`, `caltech101`, `cars`, `cifar10`, `cifar100`, `dtd`, `flowers`, `food`, `pets`, `sun397`, `voc2007`, `cub`. Most of them can be directly downloaded from torchvision, except: `caltech101`, `sun397`, `voc2007` and `cub`, which should be put under the directory of `./cache_data/raw/`.

## Training
1. Specify the model choice with `model` in the training script `run_all_datasets.sh`.
2. Use the `pretrained` parameter to provide the path to your pre-trained model.
3. Simply run the training script as follows:

   ```
   bash run_all_datasets.sh
   ```

## Acknowledgment

This part is heavily build upon [SynCLR](https://github.com/google-research/syn-rep-learn/tree/main/SynCLR). We gratefully thank the authors for their wonderful works.