# PatchCore - Notes
This is a costantly update file, with useful information and links.
## PatchCore Reimplementation
Implement PatchCore from scratch and obtain the same results of the paper.

One idea could be to implement anomaly detection and anomaly segmentation too, increasing the amount of effort (the implementation fo anomaly segmentation it's not mandatory).

PatchCore could be divided in three main tasks:
1. Pretrain on ResNet-50 on ImageNet: this is easy, download the resnet and run, not too much effort. We also have to implement the k-NN aggregation, but use PyTorch
2. Coreset subsampling: geometric and arithmetic passage to do the coreset
3. Anomaly detection and segmentation: understand how we can implement the segmentation, the paper doesn't specify it

We decided to split the work in this way:
* Aggregation of local patch features in a memory bank: Jonas
* Coreset Subsampling: Damiano
* Anomaly detection and segmentation: Sofia

## PatchCore Extension
We can decide what type of extension we'll do, but I suggest to implement everything and then combine some type of implementation to obtain better results.
### Different Pre-Training
#### Use CLIP as Pre-Training
### Network Finetuning
#### Rotation recognition task
#### Constrative learning with syntethic anomalies obtained with CutMix


## Useful Links
* [Official PatchCore implementation](https://github.com/rvorias/ind_knn_ad)
* [Another unofficial implementation](https://github.com/amazon-science/patchcore-inspection)
* [Template for the report](https://www.overleaf.com/latex/templates/cvpr-2022-author-kit/qbmjsdxryffn)