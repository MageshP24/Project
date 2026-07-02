import timm
import torch.nn as nn

def build_model(name: str, num_classes: int = 2):
    if name == "effnet_b3":
        m = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)
    elif name == "vit_b16":
        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m
