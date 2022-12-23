Good morning to everyone, I'm Damiano Ferla and I'm going to present: "Towards Total Recall in Industrial Anomaly Detection", a paper presented during the CVPR 2022.

In this paper, the authors deals with the cold-start problem: fit a model only using nominal image examples, so without defects.

They propose PatchCore, which use a memory bank of nominal patch-features. It offers competitive inference time while it obtains state-of-the-art performance for the anomaly detection and localization.

There was some past works about cold-start problem, like SPADE and PaDiM.

PatchCore is characterized by different parts: local patch features aggregated into a memory bank, a core-set reduction to increase the efficency and the algorithm to detection and localization.

**Local patch features aggregated into a memory bank**

PatchCore uses a pre-trained network on ImageNet *fi*.

One choice for the feature representation could be the last level in the features hierarchy. But this introduce two problem: it lose some nominal information and the very deep and abstract features coming from it, suffer from bias towards natural image classification task because they have a little overlap with cold-start industrial anomaly detection task.

For this reason, they proposed a memory bank of patch-level features, in order to use them during test time, avoiding features too much generalistics or too much affected by bias towards ImageNet classification.

They decided to use a local neighbourhood aggregation, in order to increase the receptive field size and robustness to little spatial deviation.

**Core subsampling**

In order to increase the sample set, the memory bank M became too much big and with it the time to evaluate new test samples and the amount of storage. This problem was already noted in SPADE during the anomaly segmentation, which use low and high level feature maps.

Random sub-sampling lose significative information in the encoded memory bank. In this work they used a coreset sub-sampling to reduce M. They also discovered that this solution reduce inference time while maintaining the perfomance.

PatchCore use a nearest neighbour computation, so we use the *minimax facility location* to select the coreset, in order to grant a similar coverage respect the original memory bank M.

The computation of the approximated memory bank M is an NP-Hard problem, so we use a greedy iterative approximation.

**Anomaly Detection**

With the patch-feature memory bank M, they estimate the anomaly level s for an image test as the maximum score of distance s* between test patch-features and each nearest neighbour m* associated to it.

To obtain s, we scale the weight w on s* to account the behaviour of neighbourhood patches. If memory bank features are closer to anomaly candidate, they are far from neighbouring samples, so we increase the anomaly score.

## Evaluation Metrics
To measure segmentation performance, we use AUROC pixel-wise and PRO, which keep in account the overlap and the recovery of anomalies connected to the components.

**Anomaly Detection on MVTec AD**

We notice that an error reduction from 2.1% of PaDiM to 0.9% of PatchCore-25%, it means an error reduction of 57%. In an industrial context, it's very significative.

With PatchCore, less than 50 images were without classification. In addiction to that, PatchCore obtains the best performance regarding anomaly segmentation, measured by AUROC pixel-wise and PRO.


**Inference Time**

The inference times included also the forward pass through the backbone.

Inference time of PatchCore-100% is lower than SPADE, but with better perfomance.

But with coreset subsampling, PatchCore could be faster, with inference time lower than PaDiM.

## Conclusions

The main results is the state-of-the-art performance about the cold-start anomaly detection and localization systems with a lower computational cost.

We know that industrial anomaly detection is one of the most successful application in the computer vision world, the improvement obtained by PatchCore could be important for who live in this domain.

Even if the main approach we used it could be used in detection system which lives in more controversial domain, we don't think the improvement is enough signficative to change actual application.

While PatchCore shows high efficiency for the industrial anomaly detection without the need to adapt to the domain of the problem, the applicability is limited to the trasferability of pre-trained features. This could be an extension work in the future.





