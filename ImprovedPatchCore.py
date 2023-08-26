import clip
class PatchCore(torch.nn.Module):
  def __init__(self, backbone: 'resnet50', out_indices: Tuple = (2,3),
               input_size: int = 224, patch_size: int = 3, stride: int = 1,
               k: int = 3, sigma: int = 4, perc_coreset: float = 0.25, eps: float =0.9):
    super().__init__()
    self.out_indices = out_indices
    self.input_size = input_size
    self.k = k                                                                          # HYPERPARAMETER
    self.sigma = sigma
    self.kernel_size = int(2 * self.sigma * 4 + 1)   
    if torch.cuda.is_available():
        self.device = 'cuda'
        self.to(self.device)
    else:
        self.device = 'cpu'
    self.feature_extractor = clip.load(backbone, device=self.device)
        #TODO:
    #for param in self.feature_extractor.parameters():
    #  param.requires_grad = False
    #self.feature_extractor.eval()                                                       # Inference mode instead of training
    self.pooling = torch.nn.AvgPool2d(patch_size, stride)                               # Pooling done after the feature extraction
    self.memory_bank = []
    self.perc_coreset = perc_coreset                                                    # HYPERPARAMETER
    self.eps = eps                                                                      # HYPERPARAMETER

  def forward(self, input: tensor):
    input = input.to(self.device)
    with torch.no_grad():
      #TODO:feature_maps = self.feature_extractor(input)
    return feature_maps

  def fit(self, input: DataLoader):
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
    self.memory_bank = patches[self.coreset_reduction(reduced_patches)]

  def predict(self, sample):
    feature_maps = self(sample)
    resized_features, feature_map_size = self.patch_extraction(feature_maps)
    min_distances, nearest_neighbor_indexes = self.nearest_neighbour_search(resized_features, 1)    # Nearest Neighbour Search
    min_distances = min_distances.squeeze()
    nearest_neighbor_indexes = nearest_neighbor_indexes.squeeze()
    max_index = torch.argmax(min_distances)
    m_test = resized_features[max_index].unsqueeze(0)                                                   # Test features
    m_star = self.memory_bank[nearest_neighbor_indexes[max_index]].unsqueeze(0)                         # Memory bank features
    s_star = torch.cdist(m_test, m_star)
    _, nb_indexes = self.nearest_neighbour_search(m_star, self.k)
    nb_features = self.memory_bank[nb_indexes]
    nb_distances = torch.cdist(m_test, nb_features)
    w = 1 - (torch.exp(s_star)/torch.sum(torch.exp(nb_distances)))
    anomaly_score = w * s_star
    segmentation_map = min_distances.reshape(1, 1, *feature_map_size)
    segmentation_map = F.interpolate(segmentation_map, size=(self.input_size, self.input_size), mode='bilinear')
    segmentation_map = transforms.functional.gaussian_blur(segmentation_map, self.kernel_size, sigma = self.sigma)

    return anomaly_score, segmentation_map

  def evaluate(self, input: DataLoader):
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

    fpr, tpr, thresholds = roc_curve(masks, segmentation_maps_flattened)    # Roc curve for segmentation map
    roc_auc_sm = auc(fpr, tpr)

    y_true = np.array(labels)
    y_scores = np.array(anomaly_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)                      #Roc curve for anomaly score
    roc_auc_ad = auc(fpr, tpr)
    
    return roc_auc_ad, roc_auc_sm, sum(inference_times)/len(inference_times)


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
    distances = torch.cdist(sample_features, self.memory_bank)
    scores, nearest_neighbor_indexes = distances.topk(k, largest=False)
    return scores, nearest_neighbor_indexes
  
  def patch_extraction(self, feature_maps):
    features = []
    for feature_map in feature_maps:
        features.append(self.pooling(feature_map))
    feature_map_size = feature_maps[0].shape[-2:]
    resized_features = self.resize(features, feature_map_size)                        # Custom methods to resize and reshape the patches
    resized_features = self.reshape(resized_features)
    return resized_features, feature_map_size
