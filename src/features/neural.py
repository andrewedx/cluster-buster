from __future__ import annotations

import numpy as np
import torch
from torchvision import models, transforms
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


@torch.inference_mode()
def compute_dinov2_descriptors(
    base_images: list[dict],
    *,
    batch_size: int = 32,
    device: str | None = None,
    model_name: str = "dinov2_vits14",  # other common: dinov2_vitb14, dinov2_vitl14
    image_size: int = 224,
) -> np.ndarray:
    """
    Compute DINOv2 descriptors from `base_images` produced by image_loader().

    Expects base_images[i]["data"] as float32 RGB in [0,1], shape (H, W, 3).

    Returns: np.ndarray of shape (N, D), float32
      - D depends on model: vits14=384, vitb14=768, vitl14=1024, vitg14=1536 (approx)
    """
    if base_images is None or len(base_images) == 0:
        return np.empty((0, 0), dtype=np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load DINOv2 model (will download weights if not cached)
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)

    # 2) Preprocess: resize/crop + ImageNet normalization (DINOv2 uses ImageNet stats)
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

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
            t = torch.from_numpy(x).permute(2, 0, 1)  # (3, H, W), float32 in [0,1]
            t = preprocess(t)  # -> (3, image_size, image_size), normalized
            batch_tensors.append(t)

        batch = torch.stack(batch_tensors, dim=0).to(device)  # (B, 3, S, S)

        # 3) Forward to get a single embedding per image
        if hasattr(model, "forward_features"):
            y = model.forward_features(batch)
            # Typical key: "x_norm_clstoken" => (B, D)
            if isinstance(y, dict) and "x_norm_clstoken" in y:
                out = y["x_norm_clstoken"]
            elif isinstance(y, dict) and "x_norm_patchtokens" in y:
                # fallback: mean pool patch tokens -> (B, D)
                out = y["x_norm_patchtokens"].mean(dim=1)
            else:
                # last resort: try taking y directly if it's already a tensor
                out = y if torch.is_tensor(y) else None
                if out is None:
                    raise RuntimeError("Unexpected DINOv2 forward_features() output format.")
        else:
            # some variants support direct forward that returns (B, D)
            out = model(batch)

        feats.append(out.detach().cpu().float().numpy())

    return np.concatenate(feats, axis=0).astype(np.float32)
