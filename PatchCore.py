import torch
import timm
import torch.nn.functional as F
import time
import numpy as np
import clip

from tqdm import tqdm
from sklearn import random_projection
from sklearn.metrics import roc_curve, auc
from torch import tensor, Tensor
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from dataset import _convert_image_to_rgb

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])


class PatchCoreBase(torch.nn.Module):
  ''' 
    PatchCore super-class.
    Parameters:
    - out_indices: Tuple that represent the output indices for the feature extractor,
    - input_size: Integer that represents the dimensions of the input image,
    - patch_size: Integer that represents the size of the window
      in the average pooling,
    - stride: Integer that represents the stride of the window in the average pooling,
    - k: Number of nearest patch-features analyzed,
    - sigma: Kernel width of the Gaussian,
    - perc_coreset: Percentage to which the original memory bank has been subsampled to,
    - eps: Parameter to control the quality of the embedding according to the Johnson-Lindenstrauss lemma.
  '''
  def __init__(self, out_indices: Tuple = (2,3),
               input_size: int = 224,
               patch_size: int = 3,
               stride: int = 1,
               k: int = 3,
               sigma: int = 4,
               perc_coreset: float = 0.25,
               eps: float =0.9):
    super().__init__()
    self.out_indices = out_indices
    self.input_size = input_size
    self.k = k                                                                          # HYPERPARAMETER
    self.sigma = sigma
    self.kernel_size = int(2 * self.sigma * 4 + 1) 
    self.feature_extractor = None
    if torch.cuda.is_available():
        self.device = 'cuda'
        self.to(self.device)
    else:
        self.device = 'cpu'                                                       # Inference mode instead of training
    self.pooling = torch.nn.AvgPool2d(patch_size, stride)                               # Pooling done after the feature extraction
    self.memory_bank = []
    self.perc_coreset = perc_coreset                                                    # HYPERPARAMETER
    self.eps = eps
    self.transformation = None
    self.target_transformation = None                                                              # HYPERPARAMETER

  '''
    Defines the computation performed at every call.
  '''
  def forward(self, input: tensor):
    raise NotImplementedError
  
  '''
    Returns the transformations to be applied to the train and target data.
  '''
  def get_transform(self):
    return self.transformation, self.target_transformation

  '''
    Training method. In PatchCore the training phase is represented by:
    - local patch features aggregated into a memory bank
    - corset-reduction to increase efficiency. To further reduce coreset selection time, 
      making use of the Johnson-Lindenstrauss theorem to reduce dimensionality of each element.
  '''
  def fit(self, input: DataLoader):
    print('Training model...')
    patches = []
    for sample, _ in tqdm(input):
      feature_maps = self(sample)
      resized_features, _ = self.patch_extraction(feature_maps)
      patches.append(resized_features)  
    patches = torch.cat(patches)
    try:                                                                                # Applying JL Theorem
      transformation = random_projection.SparseRandomProjection(eps=self.eps)
      if self.device == 'cuda':
        patches = patches.to("cpu")
      reduced_patches = torch.tensor(transformation.fit_transform(patches))
      if self.device == 'cuda':
        reduced_patches = reduced_patches.to(self.device)
        patches = patches.to(self.device)
    except ValueError:
      print(f'Error in SparseRandomProjection')
    print('Coreset reduction...')
    self.memory_bank = patches[self.coreset_reduction(reduced_patches)]

  '''
    Return anomaly detection score and relative segmentation map for a single test sample.
    Anomaly detection steps:
    - Create a locally aware patch feature of the test sample.
    - Compute the image-level anomaly detection score for the test sample by comparing
      the test patch with the nearest neighbours patches inside the memory bank.
    - Compute a segmentation map by realigning computed path anomaly scores based on
      their respective spacial location and upscale the result by bi-linear interpolation 
      and smooth the result with a gaussian blur.
  '''
  def predict(self, sample):
    feature_maps = self(sample)
    resized_features, feature_map_size = self.patch_extraction(feature_maps)
    min_distances, nearest_neighbor_indexes = self.nearest_neighbour_search(resized_features, 1)    # Nearest Neighbour Search
    min_distances = min_distances.squeeze()
    nearest_neighbor_indexes = nearest_neighbor_indexes.squeeze()
    max_index = torch.argmax(min_distances)
    m_test = resized_features[max_index].unsqueeze(0)                                                   # Test features
    m_star = self.memory_bank[nearest_neighbor_indexes[max_index]].unsqueeze(0)                         # Memory bank features
    s_star = torch.cdist(m_test.float(), m_star.float())
    _, nb_indexes = self.nearest_neighbour_search(m_star, self.k)
    nb_features = self.memory_bank[nb_indexes]
    nb_distances = torch.cdist(m_test.float(), nb_features.float())
    w = 1 - (torch.exp(s_star)/torch.sum(torch.exp(nb_distances)))
    anomaly_score = w * s_star
    segmentation_map = min_distances.reshape(1, 1, *feature_map_size)
    segmentation_map = F.interpolate(segmentation_map, size=(self.input_size, self.input_size), mode='bilinear')
    segmentation_map = transforms.functional.gaussian_blur(segmentation_map, self.kernel_size, sigma = self.sigma)

    return anomaly_score, segmentation_map

  '''
    Evaluation of the model's performance through the roc auc metric.
    This method returns:
    - false positive rate for images segmentation
    - true positive rate for images segmentation
    - false positive rate for images predictions
    - true positive rate for images predictions
    - roc_auc score for images predictions
    - roc_auc score for images segmentation
    - mean inference time
  '''
  def evaluate(self, input: DataLoader):
    print('Evaluation started...')
    anomaly_scores = []
    segmentation_maps_flattened = []
    labels = []
    masks = []
    inference_times = []
    for sample, label, mask in tqdm(input):
      start_time = time.time()
      anomaly_score, segmentation_map = self.predict(sample)
      end_time = time.time()
      inference_times.append(end_time - start_time)
      segmentation_map = segmentation_map.to("cpu")
      anomaly_scores.append(anomaly_score.item())
      labels.append(label.item())
      segmentation_maps_flattened.extend(segmentation_map.flatten().numpy())
      mask = torch.mean(mask, dim=1, keepdim=True)
      masks.extend(mask.flatten().numpy().astype(int))

    fpr_sm, tpr_sm, thresholds = roc_curve(masks, segmentation_maps_flattened)    # Roc curve for segmentation map
    roc_auc_sm = auc(fpr_sm, tpr_sm)

    y_true = np.array(labels)
    y_scores = np.array(anomaly_scores)
    fpr_as, tpr_as, thresholds = roc_curve(y_true, y_scores)                      #Roc curve for anomaly score
    roc_auc_as = auc(fpr_as, tpr_as)

    return fpr_sm, tpr_sm, fpr_as, tpr_as, roc_auc_as, roc_auc_sm, sum(inference_times)/len(inference_times)

  '''
    Custom methods to resize the patches, using adaptive average pooling.
  '''
  def resize(self, input_features: List[Tensor], new_size) -> Tensor:
    resized_features = []
    for input_feature in input_features:
      resized_features.append(F.adaptive_avg_pool2d(input_feature, new_size))
    resized_features = torch.cat(resized_features, dim=1)
    return resized_features

  '''
    Custom methods to reshape the patches
  '''
  def reshape(self, input_features: Tensor) -> Tensor:
    num_features = input_features.size(1)
    input_features = input_features.permute(0,2,3,1).reshape(-1, num_features)
    return input_features

  '''
    Greedy coreset subsampling, using the minimax facility locations 
    as a metric to select the coreset.
    Returns coreset indexes for given memory_bank.
  '''
  def coreset_reduction(self, patches: List[Tensor]) -> List[Tensor]:
    coreset_indexes = []
    index = 0
    last_item = patches[index : index + 1]
    coreset_indexes.append(index)
    min_distances = torch.linalg.norm(patches - last_item, dim=1, keepdims=True)
    while len(coreset_indexes) <= int(len(patches) * self.perc_coreset):
      distances = torch.linalg.norm(patches - last_item, dim=1, keepdims=True)
      min_distances = torch.minimum(distances, min_distances)
      index = torch.argmax(min_distances)
      last_item = patches[index : index + 1]
      min_distances[index] = 0
      coreset_indexes.append(index)
    return coreset_indexes
  
  '''
    Nearest neighbour search method.
    Parameters:
    - sample_features: test patch-feature whose k neighbors you want to find,
    - k: number of patch-features closest in distance to the given one.

  '''
  def nearest_neighbour_search(self, sample_features, k=1):
    distances = torch.cdist(sample_features, self.memory_bank)
    scores, nearest_neighbor_indexes = distances.topk(k, largest=False)
    return scores, nearest_neighbor_indexes
  
  '''
    Patch extraction method.
  '''
  def patch_extraction(self, feature_maps):
    features = []
    for feature_map in feature_maps:
        features.append(self.pooling(feature_map))
    feature_map_size = feature_maps[0].shape[-2:]
    resized_features = self.resize(features, feature_map_size) # Custom methods to resize and reshape the patches
    resized_features = self.reshape(resized_features)
    return resized_features, feature_map_size

class PatchCore(PatchCoreBase):
  '''
    PatchCore is a subclass of PatchCoreBase class. 
    The __init__ method override the one from the PatchCoreBase class with one mone parameter:
    - backbone: String that represents the model name that will be used as the backbone network
      for feature extractor. 
  '''
  def __init__(self, backbone: 'resnet50', 
               out_indices: Tuple = (2,3),
               input_size: int = 224,
               patch_size: int = 3,
               stride: int = 1,
               k: int = 3,
               sigma: int = 4,
               perc_coreset: float = 0.25,
               eps: float =0.9):
    super().__init__(out_indices, input_size,
                     patch_size, stride, k, sigma, perc_coreset, eps)
    
    self.feature_extractor = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=self.out_indices)
    for param in self.feature_extractor.parameters():
      param.requires_grad = False
    self.feature_extractor.eval()                       # Inference mode instead of training                                                                      # HYPERPARAMETER

    self.transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    self.target_transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
  '''
    Feature extraction of the input sample using feature extractor network.
    Return the extracted feature maps.
  '''
  def forward(self, input: tensor):
    input = input.to(self.device)
    with torch.no_grad():
      feature_maps = self.feature_extractor(input)
    return feature_maps

class PatchCoreWithCLIP(PatchCoreBase):
  '''
    PatchCoreWithCLIP is a subclass of PatchCoreBase class. 
    It represents an extension of the PatchCore method that exploits the pretrained Image Encoder
    of CLIP instead of the ImageNet pretrained one.
    The __init__ method override the one from the PatchCoreBase class with one mone parameter:
    - backbone: String that represents the CLIP model name that will be used as the backbone network
      for feature extractor. 
  '''
  def __init__(self, backbone: 'resnet50', 
               out_indices: Tuple = (2,3),
               input_size: int = 224, 
               patch_size: int = 3, 
               stride: int = 1,
               k: int = 3, 
               sigma: int = 4, 
               perc_coreset: float = 0.25, 
               eps: float =0.9):
    super().__init__(out_indices, input_size, 
                     patch_size, stride, k, sigma, perc_coreset, eps)
    self.feature_extractor, self.transformation = clip.load(backbone, device = self.device)
    for param in self.feature_extractor.parameters():
      param.requires_grad = False
    self.feature_extractor.eval()      # Inference mode instead of training
    
    self.target_transformation = transforms.Compose([
          Resize(size=input_size, interpolation=InterpolationMode.BICUBIC),
          CenterCrop(size=(input_size, input_size)),
          _convert_image_to_rgb ,
          ToTensor()  ])

  '''
     Feature extraction of the input sample using feature extractor network
     The custom hooks extract the layer 2 and 3 feature maps.
     Return the extracted feature maps.
  '''                                            
  def forward(self, input: tensor):
    input = input.to(self.device)
    feature_maps = []
    def custom_hook(module, input, output):
      feature_maps.append(output)
    with torch.no_grad():
      self.feature_extractor.visual.layer2[-1].register_forward_hook(custom_hook)
      self.feature_extractor.visual.layer3[-1].register_forward_hook(custom_hook)
      self.feature_extractor.encode_image(input)
    return feature_maps