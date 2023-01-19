import torch
from PatchFeatures import PatchCore
from MVTec import MVTecData
POSSIBLE_CLASSES = [
    "bottle_copy",
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


def run():
    products = ["bottle_copy", "cable"]
    for prd in products:
        # create new model for each product
        model = PatchCore(
            backbone_name='resnet50',
            out_indices=(2, 3),
            patch_size=3,
            stride=1
        )
        train_data, test_data = MVTecData(product=prd, img_size=224).return_datasets()
        model.fit(train_data)
        model.evaluate(test_data)

if __name__ == "__main__":
    run()
