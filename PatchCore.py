import torch
import timm
import torch.nn.functional as F
import time
import numpy as np

from tqdm import tqdm
from sklearn import random_projection
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch import tensor, Tensor
from typing import Tuple, List
from torch.utils.data import DataLoader

class PatchCore(torch.nn.Module):
  def __init__(self, backbone: 'resnet50', out_indices: Tuple = (2,3),
               input_size: int = 224, patch_size: int = 3, stride: int = 1,
               k: int = 3, sigma: int = 4, perc_coreset: float = 0.25, eps: float =0.9):
    super().__init__()
    self.out_indices = out_indices
    self.input_size = input_size
    self.k = k
    self.sigma = sigma
    self.kernel_size = int(2 * self.sigma * 4 + 1) #Da controllare
    if torch.cuda.is_available():
        self.device = 'cuda'
        self.to(self.device)
    else:
        self.device = 'cpu'
    self.feature_extractor = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=self.out_indices)
    # Non deve essere allenata -> Non modifico i gradienti
    for param in self.feature_extractor.parameters():
      param.requires_grad = False
    # Setta in inference mode invece che in training
    self.feature_extractor.eval()
    self.pooling = torch.nn.AvgPool2d(patch_size, stride)
    self.memory_bank = []
    self.perc_coreset = perc_coreset
    self.eps = eps

  def forward(self, input: tensor):
    input = input.to(self.device)
    with torch.no_grad():
      feature_maps = self.feature_extractor(input)
    return feature_maps

  def fit(self, input: DataLoader):
    patches = []
    for sample, _ in tqdm(input):
      # Chiamo forward()
      feature_maps = self(sample)
      # Sample shape (batch_size:1, #channels:3, height:224, width:224)
      # Faccio pooling -> Chiamo il pooling per ogni elemento della feature map
      features = []
      for feature_map in feature_maps:
        features.append(self.pooling(feature_map))
      # Faccio dei resize per ottenere le patches
      feature_map_size = feature_maps[0].shape[-2:]
      resized_features = self.resize(features, feature_map_size)
      resized_features = self.reshape(resized_features)
      patches.append(resized_features)  # LOCALLY AWARE PATCH FEATURES
    patches = torch.cat(patches)
    # Applico il teorema di Johnson-Lindenstrauss per ridurre ancora di più la dimensionalità
    try:
      transformation = random_projection.SparseRandomProjection(eps=self.eps)
      if self.device == 'cuda':
        patches = patches.to("cpu")
      reduced_patches = torch.tensor(transformation.fit_transform(patches))
      if self.device == 'cuda':
        reduced_patches = reduced_patches.to(self.device)
        patches = patches.to(self.device)
    except ValueError:
      print(f'Error in SparseRandomProjection')
    # Coreset subsampling -- ho patches[]
    self.memory_bank = patches[self.coreset_reduction(reduced_patches)]

  def predict(self, sample):
    # Questa è uguale a fit (stiamo estraendo le patch)
    feature_maps = self(sample)
    features = []
    for feature_map in feature_maps:
      features.append(self.pooling(feature_map))
    feature_map_size = feature_maps[0].shape[-2:]
    resized_features = self.resize(features, feature_map_size)
    resized_features = self.reshape(resized_features)
    # Nearest Neighbour Search
    min_distances, nearest_neighbor_indexes = self.nearest_neighbour_search(resized_features, 1)
    min_distances = min_distances.squeeze()
    nearest_neighbor_indexes = nearest_neighbor_indexes.squeeze()
    # Trova la patch con la distanza più grande dal suo nn
    max_val = torch.max(min_distances)
    max_index = torch.argmax(min_distances)
    # Prendo adesso le patch che ho trovato
    m_test = resized_features[max_index].unsqueeze(0) #features di test
    m_star = self.memory_bank[nearest_neighbor_indexes[max_index]].unsqueeze(0) #features del memory bank
    # Calcolo la norma tra m_test ed m_star
    s_star = torch.cdist(m_test, m_star)
    # TODO: Controllare se s_start sia uguale a max_val
    _, nb_indexes = self.nearest_neighbour_search(m_star, self.k)
    nb_features = self.memory_bank[nb_indexes]
    nb_distances = torch.cdist(m_test, nb_features)
    w = 1 - (torch.exp(s_star)/torch.sum(torch.exp(nb_distances)))
    anomaly_score = w * s_star  # FORMULA 7
    #print(f'----------Anomaly score: {anomaly_score}---------------')
    segmentation_map = min_distances.reshape(1, 1, *feature_map_size)
    segmentation_map = F.interpolate(segmentation_map, size=(self.input_size, self.input_size), mode='bilinear')
    segmentation_map = transforms.functional.gaussian_blur(segmentation_map, self.kernel_size, sigma = self.sigma)

    return anomaly_score, segmentation_map

  def evaluate(self, input: DataLoader):
    anomaly_scores = []
    segmentation_maps = []
    segmentation_maps_flattened = []
    labels = []
    masks = []
    masks2 = []
    inference_times = []
    for sample, label, mask in tqdm(input):
      start_time = time.time()
      anomaly_score, segmentation_map = self.predict(sample)
      end_time = time.time()
      inference_times.append(end_time - start_time)
      segmentation_map = segmentation_map.to("cpu")
      anomaly_scores.append(anomaly_score.item())
      labels.append(label.item())
      segmentation_maps.append(segmentation_map)
      segmentation_maps_flattened.extend(segmentation_map.flatten().numpy())
      mask = torch.mean(mask, dim=1, keepdim=True)
      masks2.append(mask)
      masks.extend(mask.flatten().numpy().astype(int))

    print(f'Tempo di inferenza minimo: {min(inference_times)}')
    print(f'Tempo di inferenza medio: {sum(inference_times)/len(inference_times)}')

    # CURVA ROC PER LA MAPPA DI SEGMENTAZIONE
    fpr, tpr, thresholds = roc_curve(masks, segmentation_maps_flattened)
    print(f'fpr len: {len(fpr)}')
    print(f'tpr len: {len(tpr)}')
    print(f'fpr trovati: {fpr}')
    print(f'tpr trovati: {tpr}')
    print(f'Soglia trovata: {thresholds}')
    #Scegliamo la miglior combinazione delle due tra fpr e tpr
    differences = tpr - fpr
    index = np.argmax(differences)
    best_fpr, best_tpr, best_sm_treshold = fpr[index], tpr[index], thresholds[index+2]
    print(f'Parametri trovati: I MIGLIORI POSSIBILI {best_fpr, best_tpr, best_sm_treshold}')

    roc_auc = auc(fpr, tpr)
    # Visualizzazione della curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasso di falso positivo (1 - Specificità)')
    plt.ylabel('Tasso di vero positivo (Sensibilità)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()


    # CURVA ROC PER L'ANOMALY DETECTION
    print(f'Labels: {labels}. Anomaly Scores: {anomaly_scores}')
    y_true = np.array(labels)
    y_scores = np.array(anomaly_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    print(f'fpr trovati: {fpr}')
    print(f'tpr trovati: {tpr}')
    print(f'Soglia trovata: {thresholds}')

    # ALGORITMO PER PRENERDERE IL MASSIMO DEL TPR, IL MINIMO DEL TPR E PRENDERE QUELLI CON LO STESSO INDICE
    differences = tpr - fpr
    index = np.argmax(differences)
    best_fpr, best_tpr, best_class_treshold = fpr[index], tpr[index], thresholds[index]
    print(f'Parametri trovati: I MIGLIORI POSSIBILI {best_fpr, best_tpr, best_class_treshold}')

    roc_auc = auc(fpr, tpr)
    # Visualizzazione della curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasso di falso positivo (1 - Specificità)')
    plt.ylabel('Tasso di vero positivo (Sensibilità)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()


  def resize(self, input_features: List[Tensor], new_size) -> Tensor:
    resized_features = []
    for input_feature in input_features:
      resized_features.append(F.adaptive_avg_pool2d(input_feature, new_size))
    resized_features = torch.cat(resized_features, dim=1)
    return resized_features

  def reshape(self, input_features: Tensor) -> Tensor:
    num_features = input_features.size(1)
    input_features = input_features.permute(0,2,3,1).reshape(-1, num_features)
    return input_features

  def coreset_reduction(self, patches: List[Tensor]) -> List[Tensor]:
    print('Inizio coreset reduction.')
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

  def nearest_neighbour_search(self, sample_features, k=1):
    # Calcolo la distanza fra le patch features e quelle presenti nel memory bank
    distances = torch.cdist(sample_features, self.memory_bank)
    scores, nearest_neighbor_indexes = distances.topk(k, largest=False)
    return scores, nearest_neighbor_indexes
