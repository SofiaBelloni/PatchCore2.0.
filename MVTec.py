import os
import PIL
import torch
from torch import tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATA_PATH = './data'
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])

MVTEC_CLASSES = [
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
    def __init__(self, product: str, img_size: int = 224):
        self.product = product
        self.img_size = img_size
        if product in MVTEC_CLASSES:
            self.load_data()
        else:
            print(f"{product} not part of MvTec Classes")
        # create train and test dataset
        self.train_dataset = MVTecTrain(product, img_size)
        self.test_dataset = MVTecTest(product, img_size)

    def downloadClass(self, product: str):
        if not os.path.isdir(os.path.join(DATA_PATH, product)):
            print(f"Class {product} not found. Start download")
            # TODO implement download from links see seperate links on
            #  https://www.mvtec.com/company/research/datasets/mvtec-ad

    def return_datasets(self):
        return self.train_dataset, self.test_dataset


class MVTecTrain(ImageFolder):
    def __init__(self, product: str, img_size: int = 224):
        super.__init__(
            root=os.path.join(DATA_PATH, product, "train", "good"),
            transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        self.product = product
        self.img_size = img_size


class MVTecTest(ImageFolder):
    def __init__(self, product: str, img_size: int = 224):
        super.__init__(
            root=os.path.join(DATA_PATH, product, "test"),
            transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        self.product = product
        self.size = img_size

    #def __getitem__(self, item):
        #TODO implent get item with distinction between good vs rest in path
