# 1,2,3,4 eingabe -> to choose output layer of (wide) resnet 50
from typing import Tuple, Dict, List
import torch
from torch import Tensor, tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import tqdm

net = timm.create_model('resnet50', pretrained=True)
net.eval()
print(net.feature_info)
# [{'num_chs': 64, 'reduction': 2, 'module': 'act1'}, {'num_chs': 256, 'reduction': 4, 'module': 'layer1'},
# {'num_chs': 512, 'reduction': 8, 'module': 'layer2'}, {'num_chs': 1024, 'reduction': 16, 'module': 'layer3'},
# {'num_chs': 2048, 'reduction': 32, 'module': 'layer4'}]

class PatchCore(torch.nn.Module):
    def __init__(
            self,
            backbone_name: str = 'resnet50',
            out_indices: Tuple = (2, 3),
            patch_size: int = 3,
            stride: int = 1,
    ):
        super.__init__()
        self.out_indices = out_indices
        self.feature_extractor = timm.create_model(backbone_name, out_indices=out_indices, pretrained=True)
        for weight in self.feature_extractor.parameters():
            weight.requires_grad = False  # no learning of resnet -> no gradients required
        self.feature_extractor.eval()  # call .eval() to set in inference mode instead of training (default)
        self.backbone_name = backbone_name
        # average function for pooling of j and j+1 backbone layer
        self.feature_avg_pooler = torch.nn.AvgPool2d(kernel_size=patch_size, stride=stride)
        self.patch_lib = []

    def get_features(self, x: tensor):
        x = x.to(self.device)
        with torch.no_grad():
            feature_maps = self.feature_extractor(x)
        # feature_maps = [feature_map.to("cpu") for feature_map in feature_maps]
        return feature_maps

    def fit(self, train_dataloader):
        for sample, _ in tqdm(train_dataloader):
            # extract features from pretrained net
            feature_maps = self.get_features(sample)
            # avg pool feature patches
            features = []
            for feat_map in feature_maps:
                features.append(self.feature_avg_pooler(feat_map))
            # Resize features to layer j size
            layer_j_size = feature_maps[0].shape[-2:]
            resized_features = self.resized_embeds(features, layer_j_size)
            resized_features = self.reshape(resized_features)
            self.patch_lib.append(resized_features)

            # TODO apply patch_lib reduction here

    def evaluate (self, test_dataloader: DataLoader):
        for sample, _, _ in tqdm(test_dataloader):
            # extract features from pretrained net
            feature_maps = self.get_features(sample)
            # avg pool feature patches
            features = []
            for feat_map in feature_maps:
                features.append(self.feature_avg_pooler(feat_map))
            # Resize features to layer j size
            layer_j_size = feature_maps[0].shape[-2:]
            resized_features = self.resized_embeds(features, layer_j_size)
            resized_features = self.reshape(resized_features)

            # TODO build knn distance etc.

    @staticmethod
    def resized_embeds(features: List[Tensor], size) -> Tensor:
        """
        :param features: List of patch-features
        :param size: size of feature layer j
        :return: resized embeddings
        """
        # get feature map size of layer j
        resized_embeds = features[0]
        for fmap in features[1:]:
            emb = F.adaptive_avg_pool2d(input=fmap, output_size=size)
            resized_embeds = torch.cat((resized_embeds, emb), 1)
        return resized_embeds

    @staticmethod
    def reshape(embedding: Tensor) -> Tensor:
        """
        Reshape Embedding from
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]
        :param embedding: Embedding tensor extracted from CNN features.
        :return: Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

