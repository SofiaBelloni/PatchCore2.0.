import wget
import os
import tarfile

from torchvision.datasets import ImageFolder
from torch import tensor
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

def _convert_image_to_rgb(image):
    return image.convert("RGB")

DATA_PATH = "/content/mvtec_anomaly_detection"

POSSIBLE_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
class MVTecData(ImageFolder):
  '''
    This class represents the MVTec Dataset.
    Parameters:
    - product: string that represents one of the possible MVTec Dataset classes,
    - img_size: dimension of the images
  '''
  def __init__(self, product, transform, target_transform):
    self.product = product
    if product in POSSIBLE_CLASSES:
      url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
      if not os.path.isdir(f'{DATA_PATH}'):
        if not os.path.exists(f'{DATA_PATH}.tar.xz'):
          print('Downloading dataset...')
          wget.download(url)
        print('Extracting dataset...')
        with tarfile.open(DATA_PATH+".tar.xz") as tar:
          tar.extractall(DATA_PATH)
      self.train_dataset = MVTecTrain(product, transform )
      self.test_dataset = MVTecTest(product, transform, target_transform)
    else:
      print(f'{product} not present in MVTEC Dataset')
      
  '''
    Returns train and test datasets.
  '''
  def get_datasets(self):
    return self.train_dataset, self.test_dataset

class MVTecTrain(ImageFolder):
  '''
    This class represents the MVTec TRAIN Dataset.
    Parameters:
    - product: string that represents one of the possible MVTec Dataset classes,
    - transform: transformations to be applied to the images, 
  '''
  def __init__(self, product, transform):
    super().__init__(
        root = f'{DATA_PATH}/{product}/train/',
        transform = transform
    )
    self.product = product

class MVTecTest(ImageFolder):
  '''
    This class represents the MVTec TEST Dataset.
     Parameters:
    - product: string that represents one of the possible MVTec Dataset classes,
    - transform: transformations to be applied to the images, 
    - target transform: transformations to be applied to the target (mask).
  '''
  def __init__(self, product, transform, target_transform):
    super().__init__(
        root = f'{DATA_PATH}/{product}/test/',
        transform = transform,
        target_transform = target_transform,
    )
    self.product = product

  def __getitem__(self, index):
    path, _ = self.samples[index]
    sample = self.loader(path)
    if "good" in path:
      mask = Image.new("L", (256,256))
      sample_class = 0
    else:
      mask_path = path.replace("test", "ground_truth")
      mask_path = mask_path.replace(".png", "_mask.png")
      mask = self.loader(mask_path)
      sample_class = 1
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      mask = self.target_transform(mask)
    return sample, sample_class, mask