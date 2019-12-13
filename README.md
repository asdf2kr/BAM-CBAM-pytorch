BAM & CBAM Pytorch
==================

Pytorch implementation of BAM and CBAM.
## BAM & CBAM Pytorch

This code purpose to evaluate of popular attention model architectures, such as BAM, CBAM on the CIFAR dataset.

#### Getting Started
```bash
$ git clone https://github.com/asdf2kr/BAM-CBAM-pytorch.git
$ cd BAM-CBAM-pytorch
$ python main.py --arch bam (default: bam network based on resnet50)
```

#### Performance
The table below shows models, dataset and performances

Model | Backbone | Dataset | Top-1 | Top-5 | Size
:----:| :----:| :------:| :----:|:-----:|:----:
- | resnet50 |CIFAR-100 | 78.93% | - | 23.70M
BAM | resnet50 |CIFAR-100 | 79.62% | - | 24.06M
CBAM | resnet50 |CIFAR-100 | 81.02% | - | 26.23M

#### To-do
Simple setup readme
Add ImageNet datasets.
