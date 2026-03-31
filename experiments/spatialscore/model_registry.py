"""Lazy model loader for SpatialAgent tool models.

Loads models on first use and caches them on GPU.
Designed for RTX 3090 (24GB) — all tool models fit alongside Qwen2.5-VL-3B.
"""

import os
import sys

# Add SpatialAgent submodule paths
SPATIALSCORE_CODE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "literature", "spatialscore", "code"
)
SPATIAL_AGENT_DIR = os.path.join(SPATIALSCORE_CODE, "SpatialAgent")


class ModelRegistry:
    """Lazily loads and caches tool models on GPU."""

    def __init__(self, checkpoints_dir: str, device: str = "cuda"):
        self._cache = {}
        self._device = device
        self._checkpoints_dir = checkpoints_dir

    def _ensure_path(self, *subpaths):
        for p in subpaths:
            if p not in sys.path:
                sys.path.insert(0, p)

    def get_depth_model(self):
        """Load DepthAnythingV2 (vitl variant)."""
        import torch
        if "depth" not in self._cache:
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR, "DepthAnythingV2"))
            from depth_anything_v2.dpt import DepthAnythingV2

            model = DepthAnythingV2(
                encoder="vitl",
                features=256,
                out_channels=[256, 512, 1024, 1024],
            )
            ckpt_path = os.path.join(self._checkpoints_dir, "depth_anything_v2_vitl.pth")
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model = model.to(self._device).eval()
            self._cache["depth"] = model
        return self._cache["depth"]

    def get_raft(self):
        """Load RAFT optical flow model."""
        import torch
        if "raft" not in self._cache:
            self._ensure_path(
                os.path.join(SPATIAL_AGENT_DIR, "RAFT"),
                os.path.join(SPATIAL_AGENT_DIR, "RAFT", "core"),
            )
            from raft import RAFT
            from argparse import Namespace

            args = Namespace(
                small=False,
                mixed_precision=True,
                corr_levels=4,
                corr_radius=4,
                dropout=0.0,
                alternate_corr=False,
            )
            model = RAFT(args)
            ckpt_path = os.path.join(self._checkpoints_dir, "raft-things.pth")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Handle DataParallel state dict prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            model = model.to(self._device).eval()
            self._cache["raft"] = model
        return self._cache["raft"]

    def get_orient_model(self):
        """Load OrientAnything model + preprocessing."""
        import torch
        import torchvision.transforms as T
        if "orient" not in self._cache:
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR, "OrientAnything"))
            from vision_tower import DINOv2_MLP

            model = DINOv2_MLP(dino_mode="large", in_dim=1024, out_dim=720, evaluate=True)
            ckpt_path = os.path.join(self._checkpoints_dir, "orient_anything.pth")
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model = model.to(self._device).eval()

            val_preprocess = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._cache["orient"] = (model, val_preprocess)
        return self._cache["orient"]

    def get_sam2(self):
        """Load SAM2 image predictor."""
        if "sam2" not in self._cache:
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            config_path = os.path.join(SPATIAL_AGENT_DIR, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
            ckpt_path = os.path.join(self._checkpoints_dir, "sam2.1_hiera_large.pt")
            model = build_sam2(config_path, ckpt_path, device=self._device)
            predictor = SAM2ImagePredictor(model)
            self._cache["sam2"] = predictor
        return self._cache["sam2"]

    def get_ram(self):
        """Load RAM (Recognize Anything Model) for tagging."""
        if "ram" not in self._cache:
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR))
            from ram.models import ram_plus
            import torchvision.transforms as T

            model = ram_plus(
                pretrained=os.path.join(self._checkpoints_dir, "ram_plus_swin_large_14m.pth"),
                image_size=384,
                vit="swin_l",
            )
            model = model.to(self._device).eval()

            transform = T.Compose([
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._cache["ram"] = (model, transform)
        return self._cache["ram"]

    def get_vggt(self):
        """Load VGGT model for camera parameter estimation."""
        if "vggt" not in self._cache:
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR))
            # VGGT loading is complex — use HuggingFace model if available
            from vggt.models.vggt import VGGT as VGGTModel

            model = VGGTModel.from_pretrained("facebook/VGGT-1B")
            model = model.to(self._device).eval()
            self._cache["vggt"] = model
        return self._cache["vggt"]
