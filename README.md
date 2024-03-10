# LAAT

The official implementation of LAAT ([Language-Driven Anchors for Zero-Shot Adversarial Robustness](https://arxiv.org/abs/2301.13096), CVPR 2024)

## Prerequisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python == 3.9.12
- torch == 1.12.1
- torchvision == 0.13.1
- [autoattack](https://github.com/fra31/auto-attack) == 0.1

## Usage

### Craft anchors

To craft anchors with expansion algorithm, in `anchors/`,

```bash
python get_anchors.py --classes cifarfs_classes.txt --to cifarfs_anchors.npy
python convert_anchor.py cifarfs 64
```

Note that 64 means the first 64 classes are training classes, others are testing classes, which are not used to construct expansion mapping.

### Train & Test

Here we give an exmaple. 

The following command can train a zero-shot robust classidier on CIFAR-FS with the Conv4-512 backbone:

```bash
python train.py \
--data_dir /path/to/dataset --exp_name /path/to/save \
--seed 3407 \
--n_support 0 \
--dataset CIFAR100FS \
--model Conv4-512 \
--use_linear --head cos-span \
--train_type TRADES-cos \
--suffix 1
```

Test FGSM attack:

```bash
python train.py \
--data_dir /path/to/dataset --exp_name /path/to/save \
--seed 3407 \
--n_support 0 \
--dataset CIFAR100FS \
--model Conv4-512 \
--use_linear --head cos-span \
--train_type TRADES-cos \
--suffix 1 \
--eval --load_best \
--attack FGSM
```

Test AutoAttack:

```bash
python train.py \
--data_dir /path/to/dataset --exp_name /path/to/save \
--seed 3407 \
--n_support 0 \
--dataset CIFAR100FS \
--model Conv4-512 \
--use_linear --head cos-span \
--train_type TRADES-cos \
--suffix 1 \
--eval --load_best \
--attack AA --n_test 100
```

To train & test 1-shot, change `--n_support 0` to `--n_support 1`.

## Datasets

- [CIFAR-FS](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing)
- [miniImageNet](https://drive.google.com/open?id=1R6dA6QGEW-lmiNkitCwK4IkAbl4uT3y3)

### If you find this useful in your research, please cite this work:
```
@inproceedings{li2024languagedriven,
  author = {Li, Xiao and Zhang, Wei and Liu, Yining and Hu, Zhanhao and Zhang, Bo and Hu, Xiaolin},
  title = {Language-Driven Anchors for Zero-Shot Adversarial Robustness},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```
