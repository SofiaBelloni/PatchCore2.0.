# PatchCore2.0

**Anomaly Detection project**

*Advanced Machine Learning, 11 September 2023*

The purpose of the project is to familiarise itself with the anomaly detection task, focusing on the industrial
world. For this reason, the ’cold start’ problem was tackled, i.e. building a model that is capable of detecting anomalies in industrial objects, despite only
having been trained on nominal, defect-free samples.
This approach is necessary because anomalies can vary from small changes such as light scratches, to larger structural defects such as missing components.
This variety of anomalies makes it difficult to define in advance what types of problems might occur in industrial products, making it difficult to collect
datasets of abnormal samples. 
Consequently, the model is only trained on normal data and must be able to detect, during testing, all
samples with defects never shown before.

The starting point of our analysis was to implement PatchCore from scratch, trying
to obtain results similar to the original one. 
Subsequently, modifications were made to the original model, with the aim of improving performance. In particular, we replaced the pre-trained network on ImageNet with CLIP’s pretrained Image Encoder.

**Check s305953_s303393_project5.pdf for the full analysis.**
