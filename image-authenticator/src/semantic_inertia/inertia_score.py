import numpy as np
import torch
import torch.nn.functional as F


def high_frequency_energy(x: torch.Tensor) -> float:
    """
    Compute high-frequency energy using FFT.
    Input: Tensor (C, H, W) in [0,1]
    """
    if x.dim() == 3:
        x = x.mean(dim=0)  # (H, W)

    fft = torch.fft.fft2(x)
    mag = torch.abs(fft)

    h, w = mag.shape
    hf = mag[h // 4 :, w // 4 :]  # high-frequency region
    return hf.mean().item()


def semantic_inertia_features(embeddings, images):
    """
    Returns EXACTLY 5 values:
    1. Mean semantic drift
    2. Drift variance
    3. Maximum drift
    4. Drift energy
    5. Frequency instability
    """

    # -----------------------------
    # Semantic drift (cosine distance)
    # -----------------------------
    distances = []

    for i in range(len(embeddings) - 1):
        d = 1 - F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[i + 1].unsqueeze(0)
        )
        distances.append(d.item())

    if len(distances) == 0:
        mean_drift = 0.0
        var_drift = 0.0
        max_drift = 0.0
        drift_energy = 0.0
    else:
        distances = np.array(distances)
        mean_drift = float(distances.mean())
        var_drift = float(distances.var())
        max_drift = float(distances.max())
        drift_energy = float(np.sum(distances ** 2))

    # -----------------------------
    # Frequency instability
    # -----------------------------
    freq_energies = []
    for img in images:
        if isinstance(img, torch.Tensor):
            freq_energies.append(high_frequency_energy(img))

    if len(freq_energies) == 0:
        freq_instability = 0.0
    else:
        freq_instability = float(np.var(freq_energies))

    return (
        mean_drift,
        var_drift,
        max_drift,
        drift_energy,
        freq_instability
    )
