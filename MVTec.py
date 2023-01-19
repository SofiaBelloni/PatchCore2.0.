import requests, zipfile, io
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
            self.downloadClass(product=product)
        else:
            print(f"{product} not part of MvTec Classes")
        # create train and test dataset
        self.train_dataset = MVTecTrain(product, img_size)
        self.test_dataset = MVTecTest(product, img_size)

    def downloadClass(self, product: str):
        if not os.path.isdir(os.path.join(DATA_PATH, product)):
            print(f"Class {product} not found. Start download.")
            target_path = os.path.join(DATA_PATH, product)
            if product == "bottle":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370" \
                       "-1629951468/bottle_copy.tar.xz "
            elif product == "cable":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413" \
                       "-1629951498/cable.tar.xz "
            elif product == "capsule":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454" \
                       "-1629951595/capsule.tar.xz "
            elif product == "carpet":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484" \
                       "-1629951672/carpet.tar.xz "
            elif product == "grid":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487" \
                       "-1629951814/grid.tar.xz "
            elif product == "hazelnut":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545" \
                       "-1629951845/hazelnut.tar.xz "
            elif product == "leather":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607" \
                       "-1629951964/leather.tar.xz "
            elif product == "metal_nut":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637" \
                       "-1629952063/metal_nut.tar.xz "
            elif product == "pill":
                link = "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129" \
                       "-1629953099/pill.tar.xz "
            elif product == "screw":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130" \
                       "-1629953152/screw.tar.xz "
            elif product == "tile":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133" \
                       "-1629953189/tile.tar.xz "
            elif product == "toothbrush":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134" \
                       "-1629953256/toothbrush.tar.xz "
            elif product == "transistor":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166" \
                       "-1629953277/transistor.tar.xz "
            elif product == "wood":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383" \
                       "-1629953354/wood.tar.xz "
            elif product == "zipper":
                link = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385" \
                       "-1629953449/zipper.tar.xz "
            else:
                print("No download link provided.")
                return
            r = requests.get(link)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            os.mkdir(target_path)
            z.extractall(target_path)

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

mvt = MVTecData(product="bottle", img_size=224)