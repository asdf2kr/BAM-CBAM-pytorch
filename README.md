BAM & CBAM Pytorch
==================

[Pytorch](https://pytorch.org/) implementation of BAM and CBAM.
## BAM & CBAM Pytorch

This code purpose to evaluate of popular attention model architectures, such as BAM, CBAM on the CIFAR dataset.

> Park J, Woo S, Lee J Y, Kweon I S. BAM: Bottleneck Attention Module. 2018. [BMVC2018(Oral)](https://arxiv.org/pdf/1807.06514.pdf)

> Woo S, Park J, Lee J Y, Kweon I S. CBAM: Convolutional Block Attention Module. 2018. [ECCV2018](https://arxiv.org/pdf/1807.06521.pdf)

#### Architecture

BAM
![image](https://user-images.githubusercontent.com/26369382/98519653-693d1300-22b4-11eb-8f29-fd7ff2520ee5.png)

CBAM
![image](https://user-images.githubusercontent.com/26369382/98519785-9689c100-22b4-11eb-8bc6-b9fd0445f258.png)

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
ResNet| resnet50 |CIFAR-100 | 78.93% | - | 23.70M
BAM   | resnet50 |CIFAR-100 | 79.62% | - | 24.06M
CBAM  | resnet50 |CIFAR-100 | 81.02% | - | 26.23M

#### Reference
[Official PyTorch code](https://github.com/Jongchan/attention-module)
