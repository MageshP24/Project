import torch
from torchvision import transforms
from torchvision.transforms import functional as F


def get_transformations():
    """
    Meaning-preserving transformations for semantic inertia analysis.
    All transformations return TENSORS (C, H, W).
    """

    return [

        # 1️⃣ Identity (baseline)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),

        # 2️⃣ Small spatial translation
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.affine(
                    x, angle=0, translate=(3, 3), scale=1.0, shear=0
                )
            )
        ]),

        # 3️⃣ Controlled Gaussian noise
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.clamp(
                    x + 0.02 * torch.randn_like(x), 0.0, 1.0
                )
            )
        ]),

        # 4️⃣ Sub-pixel horizontal shift
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.roll(x, shifts=1, dims=2)
            )
        ]),

        # 5️⃣ JPEG-like compression simulation
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.clamp(
                    (x * 255).byte().float() / 255.0, 0.0, 1.0
                )
            )
        ])
    ]
