from __future__ import annotations

import numpy as np
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights


@torch.inference_mode()
def compute_resnet50_descriptors(
    base_images: list[dict],
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """
    Compute ResNet50 (ImageNet) descriptors from `base_images` produced by image_loader().

    Expects base_images[i]["data"] as float32 RGB in [0,1], shape (H, W, 3).

    Returns: np.ndarray of shape (N, 2048), float32.
    """
    if base_images is None or len(base_images) == 0:
        return np.empty((0, 2048), dtype=np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # output becomes 2048-d embedding
    model.eval().to(device)

    preprocess = weights.transforms()  # includes resize/crop + normalize (expects float [0,1])

    feats = []
    n = len(base_images)

    for start in range(0, n, batch_size):
        batch_items = base_images[start : start + batch_size]

        batch_tensors = []
        for i, item in enumerate(batch_items):
            x = np.asarray(item["data"])
            if x.ndim != 3 or x.shape[-1] != 3:
                raise ValueError(f"Expected HxWx3 RGB for item {start+i}, got {x.shape}")
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)

            # numpy HWC -> torch CHW
            t = torch.from_numpy(x).permute(2, 0, 1)  # (3, H, W)
            t = preprocess(t)  # -> (3, 224, 224), normalized
            batch_tensors.append(t)

        batch = torch.stack(batch_tensors, dim=0).to(device)  # (B, 3, 224, 224)
        out = model(batch)  # (B, 2048)
        feats.append(out.detach().cpu().float().numpy())

    return np.concatenate(feats, axis=0).astype(np.float32)