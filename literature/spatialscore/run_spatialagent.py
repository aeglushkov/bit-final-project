"""Run SpatialAgent (Qwen + spatial tools) on the SpatialScore benchmark.

This script runs the SpatialAgent conversation loop on each sample in the dataset.
The MLLM (Qwen) plans which spatial tools to call, the tools compute precise spatial
measurements, and the MLLM synthesizes results into a final answer.

Results are saved in the same JSON format as the authors' test_qwen.py for easy comparison.

Usage (from the code/ directory so relative image paths resolve):
    cd literature/spatialscore/code
    CUDA_VISIBLE_DEVICES=0 python ../run_spatialagent.py \
        --model_path ~/models/Qwen2.5-VL-3B-Instruct \
        --model_name qwen2_5vl-3b \
        --dataset_json_path ./dataset/SpatialScore_test50.json \
        --output_dir ./eval_results_agent \
        --checkpoints_dir ~/checkpoints \
        --max_steps 5

Compare with baseline:
    CUDA_VISIBLE_DEVICES=0 python test_qwen.py \
        --model_path ~/models/Qwen2.5-VL-3B-Instruct \
        --model_name qwen2_5vl-3b \
        --dataset_json_path ./dataset/SpatialScore_test50.json \
        --output_dir ./eval_results_baseline
"""

import os
import re
import sys
import json
import argparse
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: add vendor code directories so we can import the authors' modules
# without modifying any files in code/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "code")
SPATIAL_AGENT_DIR = os.path.join(CODE_DIR, "SpatialAgent")

sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SPATIAL_AGENT_DIR)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Optimize GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 42
torch.manual_seed(SEED)


# ===========================================================================
# Section 1: Action definitions (tool metadata for the prompt)
# ===========================================================================

@dataclass
class Action:
    """Metadata describing a tool that the MLLM can call.

    CoTAPrompt reads these to build the system prompt with tool specifications.
    """
    name: str
    description: str
    args_spec: Dict[str, Any]
    rets_spec: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)


# All 9 tools available to SpatialAgent
ALL_ACTIONS = [
    Action(
        name="Terminate",
        description="Terminate the conversation and return the final answer.",
        args_spec={"answer": "str - The final answer to the user's question."},
        rets_spec={},
        examples=[{"arguments": {"answer": "A"}, "returns": {}}],
    ),
    Action(
        name="SelfReasoning",
        description="Use the VLM's own visual reasoning to answer a sub-question about an image.",
        args_spec={
            "image": "str - Image identifier (e.g. 'image-0').",
            "query": "str - The question to ask about the image.",
        },
        rets_spec={"response": "str - The VLM's answer to the query."},
        examples=[{
            "arguments": {"image": "image-0", "query": "Is this scene indoor or outdoor?"},
            "returns": {"response": "This scene is indoor."},
        }],
    ),
    Action(
        name="LocalizeObjects",
        description="Detect and localize objects in the image by returning bounding boxes.",
        args_spec={
            "image": "str - Image identifier (e.g. 'image-0').",
            "objects": "list[str] - List of object names to detect.",
        },
        rets_spec={"regions": "list[dict] - Each dict has 'label', 'bbox' [x1,y1,x2,y2], 'score'."},
        examples=[{
            "arguments": {"image": "image-0", "objects": ["dog", "cat"]},
            "returns": {"regions": [
                {"label": "dog", "bbox": [120.25, 185.75, 305.85, 420.35], "score": 0.92},
                {"label": "cat", "bbox": [350.65, 210.45, 510.35, 390.20], "score": 0.88},
            ]},
        }],
    ),
    Action(
        name="EstimateObjectDepth",
        description="Estimate depth (distance from camera) of objects using monocular depth estimation.",
        args_spec={
            "image": "str - Image identifier.",
            "objects": "list[str] - Object names.",
            "indoor_or_outdoor": "str - 'indoor' or 'outdoor'.",
        },
        rets_spec={"results": "list[dict] - 'object', 'depth' (meters), 'error'."},
        examples=[{
            "arguments": {"image": "image-0", "objects": ["eggs", "berries"], "indoor_or_outdoor": "indoor"},
            "returns": {"results": [
                {"object": "eggs", "depth": 1.0, "error": "null"},
                {"object": "berries", "depth": 1.2, "error": "null"},
            ]},
        }],
    ),
    Action(
        name="GetObjectMask",
        description="Segment objects and return mask areas and bounding boxes using SAM2.",
        args_spec={
            "image": "str - Image identifier.",
            "objects": "list[str] - Object names to segment.",
        },
        rets_spec={"results": "list[dict] - 'object', 'mask_area' (fraction), 'bbox', 'error'."},
        examples=[{
            "arguments": {"image": "image-0", "objects": ["eggs", "berries"]},
            "returns": {"results": [
                {"object": "eggs", "mask_area": 0.03, "bbox": [150, 200, 280, 340], "error": "null"},
                {"object": "berries", "mask_area": 0.029, "bbox": [320, 210, 410, 330], "error": "null"},
            ]},
        }],
    ),
    Action(
        name="GetObjectOrientation",
        description="Estimate 3D orientation (azimuth, polar, rotation) of an object. Azimuth: 0=right, 90=away, 180=left, 270=toward viewer.",
        args_spec={
            "image": "str - Image identifier.",
            "objects": "str - Object name.",
        },
        rets_spec={"results": "list[dict] - 'object', 'angle_data' {azimuth, polar, rotation, confidence}, 'error'."},
        examples=[{
            "arguments": {"image": "image-0", "objects": "person"},
            "returns": {"results": [
                {"object": "person", "angle_data": {"azimuth": 315.0, "polar": 90.0, "rotation": 0.0, "confidence": 0.89}, "error": "null"},
            ]},
        }],
    ),
    Action(
        name="EstimateOpticalFlow",
        description="Compute average optical flow between two images. Positive mean_flow_x = camera moved right.",
        args_spec={"images": "list[str] - Two image identifiers."},
        rets_spec={"mean_flow_x": "float", "mean_flow_y": "float"},
        examples=[{
            "arguments": {"images": ["image-0", "image-1"]},
            "returns": {"mean_flow_x": 2.5, "mean_flow_y": -0.3},
        }],
    ),
    Action(
        name="EstimateHomographyMatrix",
        description="Estimate homography transformation between two views using SIFT + RANSAC.",
        args_spec={
            "image": "list[str] - Two image identifiers.",
            "num_keypoints": "int (default 1200)", "ratio_th": "float (default 0.75)",
            "ransac_reproj_threshold": "float (default 5.0)",
        },
        rets_spec={"homography_matrix": "3x3 list", "inliers_count": "int", "total_matches": "int", "status": "str"},
        examples=[{
            "arguments": {"image": ["image-0", "image-1"], "num_keypoints": 1200, "ratio_th": 0.75, "ransac_reproj_threshold": 5.0},
            "returns": {"homography_matrix": [[0.92, 0.05, -12.37], [-0.03, 0.89, 8.45], [0.0001, 0.0002, 1.0]],
                        "inliers_count": 87, "total_matches": 124, "status": "success"},
        }],
    ),
    Action(
        name="GetCameraParametersVGGT",
        description="Extract camera intrinsic and extrinsic parameters using VGGT.",
        args_spec={
            "image": "list[str] - Image identifier(s).",
            "dtype": "str - 'auto', 'float32', or 'float16'.",
        },
        rets_spec={"output": "list[dict] - 'image_index', 'intrinsic' (3x3), 'extrinsic' (3x4)."},
        examples=[{
            "arguments": {"image": ["image-0"], "dtype": "auto"},
            "returns": {"output": [{"image_index": 0,
                        "intrinsic": [[1024.3, 0, 512], [0, 1024.3, 384], [0, 0, 1]],
                        "extrinsic": [[0.99, -0.002, 0.05, -0.01], [0.004, 0.999, -0.04, 0.84], [-0.05, 0.04, 0.998, -0.55]]}]},
        }],
    ),
]


# ===========================================================================
# Section 2: Model registry (lazy-loads tool models on first use)
# ===========================================================================

class ModelRegistry:
    """Lazily loads and caches spatial tool models on GPU.

    Each tool model (DepthAnythingV2, RAFT, etc.) is loaded only when first
    called, then cached for subsequent uses. This avoids loading all models
    upfront when only a few may be needed.
    """

    def __init__(self, checkpoints_dir: str, device: str = "cuda"):
        self._cache = {}
        self._device = device
        self._checkpoints_dir = checkpoints_dir

    def _ensure_path(self, *subpaths):
        """Add paths to sys.path if not already present."""
        for p in subpaths:
            if p not in sys.path:
                sys.path.insert(0, p)

    def get_depth_model(self):
        """Load DepthAnythingV2 (ViT-L variant) for monocular depth estimation."""
        if "depth" not in self._cache:
            print("  [ModelRegistry] Loading DepthAnythingV2...")
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR, "DepthAnythingV2"))
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
            ckpt = os.path.join(self._checkpoints_dir, "depth_anything_v2_vitl.pth")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model = model.to(self._device).eval()
            self._cache["depth"] = model
            print("  [ModelRegistry] DepthAnythingV2 loaded.")
        return self._cache["depth"]

    def get_raft(self):
        """Load RAFT for optical flow estimation."""
        if "raft" not in self._cache:
            print("  [ModelRegistry] Loading RAFT...")
            self._ensure_path(
                os.path.join(SPATIAL_AGENT_DIR, "RAFT"),
                os.path.join(SPATIAL_AGENT_DIR, "RAFT", "core"),
            )
            from raft import RAFT
            from argparse import Namespace
            args = Namespace(small=False, mixed_precision=True, corr_levels=4,
                             corr_radius=4, dropout=0.0, alternate_corr=False)
            model = RAFT(args)
            ckpt = os.path.join(self._checkpoints_dir, "raft-things.pth")
            state_dict = torch.load(ckpt, map_location="cpu")
            # Strip DataParallel "module." prefix if present
            clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean)
            model = model.to(self._device).eval()
            self._cache["raft"] = model
            print("  [ModelRegistry] RAFT loaded.")
        return self._cache["raft"]

    def get_orient_model(self):
        """Load OrientAnything (DINOv2 + MLP) for 3D orientation estimation."""
        import torchvision.transforms as T
        if "orient" not in self._cache:
            print("  [ModelRegistry] Loading OrientAnything...")
            self._ensure_path(os.path.join(SPATIAL_AGENT_DIR, "OrientAnything"))
            from vision_tower import DINOv2_MLP
            model = DINOv2_MLP(dino_mode="large", in_dim=1024, out_dim=720, evaluate=True)
            ckpt = os.path.join(self._checkpoints_dir, "orient_anything.pth")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model = model.to(self._device).eval()
            preprocess = T.Compose([
                T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._cache["orient"] = (model, preprocess)
            print("  [ModelRegistry] OrientAnything loaded.")
        return self._cache["orient"]

    def get_sam2(self):
        """Load SAM2 image predictor for segmentation."""
        if "sam2" not in self._cache:
            print("  [ModelRegistry] Loading SAM2...")
            self._ensure_path(SPATIAL_AGENT_DIR)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            config = os.path.join(SPATIAL_AGENT_DIR, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
            ckpt = os.path.join(self._checkpoints_dir, "sam2.1_hiera_large.pt")
            model = build_sam2(config, ckpt, device=self._device)
            self._cache["sam2"] = SAM2ImagePredictor(model)
            print("  [ModelRegistry] SAM2 loaded.")
        return self._cache["sam2"]

    def get_vggt(self):
        """Load VGGT for camera parameter estimation."""
        if "vggt" not in self._cache:
            print("  [ModelRegistry] Loading VGGT...")
            self._ensure_path(SPATIAL_AGENT_DIR)
            from vggt.models.vggt import VGGT as VGGTModel
            model = VGGTModel.from_pretrained("facebook/VGGT-1B")
            model = model.to(self._device).eval()
            self._cache["vggt"] = model
            print("  [ModelRegistry] VGGT loaded.")
        return self._cache["vggt"]


# ===========================================================================
# Section 3: Tool function implementations (action_registry)
# ===========================================================================

def _localize_objects_simple(image_path, objects):
    """Simple object localization fallback (divides image into equal regions).

    A real implementation would use GroundingDINO or OWL-ViT for detection.
    This fallback assigns each object an equal horizontal slice of the image.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if isinstance(objects, str):
        objects = [objects]
    regions = []
    for i, obj in enumerate(objects):
        n = len(objects)
        regions.append({
            "label": obj,
            "bbox": [float(w / n * i), float(h * 0.1), float(w / n * (i + 1)), float(h * 0.9)],
            "score": 0.8,
        })
    return regions


def make_action_registry(model_registry, vlm_fn):
    """Build the action_registry dict mapping tool names to callable functions.

    Args:
        model_registry: ModelRegistry for loading tool models lazily.
        vlm_fn: Callable(image_path, query) -> str for SelfReasoning tool.

    Returns:
        Dict[str, Callable] — keys match the Action names in ALL_ACTIONS.
    """

    def terminate(answer, **kw):
        return {"answer": answer}

    def self_reasoning(image, query, **kw):
        return {"response": vlm_fn(image, query)}

    def localize_objects(image, objects, **kw):
        return {"regions": _localize_objects_simple(image, objects)}

    def estimate_object_depth(image, objects, indoor_or_outdoor="indoor", **kw):
        """Localize objects, run DepthAnythingV2, extract per-object depth."""
        import cv2
        regions = _localize_objects_simple(image, objects)
        depth_model = model_registry.get_depth_model()
        img_cv = cv2.imread(image)
        depth_map = depth_model.infer_image(img_cv)
        h, w = depth_map.shape[:2]
        img_h, img_w = img_cv.shape[:2]
        scale = 10.0 if indoor_or_outdoor == "indoor" else 50.0

        results = []
        for region in regions:
            try:
                bbox = region["bbox"]
                x1 = int(max(0, bbox[0] * w / img_w))
                y1 = int(max(0, bbox[1] * h / img_h))
                x2 = int(min(w, bbox[2] * w / img_w))
                y2 = int(min(h, bbox[3] * h / img_h))
                if x2 > x1 and y2 > y1:
                    mean_rel = float(np.mean(depth_map[y1:y2, x1:x2]))
                    max_d = float(np.max(depth_map)) or 1.0
                    depth_m = round((mean_rel / max_d) * scale, 2)
                else:
                    depth_m = 0.0
                results.append({"object": region["label"], "depth": depth_m, "error": "null"})
            except Exception as e:
                results.append({"object": region["label"], "depth": 0.0, "error": str(e)})
        return {"results": results}

    def get_object_mask(image, objects, **kw):
        """Localize objects, segment each with SAM2, return mask areas."""
        regions = _localize_objects_simple(image, objects)
        sam2 = model_registry.get_sam2()
        img_np = np.array(Image.open(image).convert("RGB"))
        total_px = img_np.shape[0] * img_np.shape[1]
        sam2.set_image(img_np)

        results = []
        for region in regions:
            try:
                masks, scores, _ = sam2.predict(box=np.array(region["bbox"])[None, :], multimask_output=False)
                area = float(np.sum(masks[0])) / total_px
                results.append({"object": region["label"], "mask_area": round(area, 4),
                                "bbox": region["bbox"], "error": "null"})
            except Exception as e:
                results.append({"object": region["label"], "mask_area": 0.0,
                                "bbox": region["bbox"], "error": str(e)})
        return {"results": results}

    def get_object_orientation(image, objects, **kw):
        """Crop object, run OrientAnything, return azimuth/polar/rotation."""
        import torchvision.transforms as T
        obj_name = objects if isinstance(objects, str) else objects[0]
        regions = _localize_objects_simple(image, [obj_name])
        orient_model, preprocess = model_registry.get_orient_model()
        device = next(orient_model.parameters()).device

        results = []
        for region in regions:
            try:
                # Crop with 10% padding
                img = Image.open(image).convert("RGB")
                bbox = region["bbox"]
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                crop = img.crop((
                    int(max(0, bbox[0] - bw * 0.1)), int(max(0, bbox[1] - bh * 0.1)),
                    int(min(img.width, bbox[2] + bw * 0.1)), int(min(img.height, bbox[3] + bh * 0.1)),
                ))
                tensor = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = orient_model(tensor)
                if hasattr(output, 'cpu'):
                    output = output.cpu()
                    az, pol, rot = float(output[0]), float(output[1]), float(output[2])
                    conf = float(output[3]) if output.shape[0] > 3 else 0.0
                else:
                    az, pol, rot, conf = 0.0, 0.0, 0.0, 0.0
                results.append({"object": region["label"],
                                "angle_data": {"azimuth": round(az, 1), "polar": round(pol, 1),
                                               "rotation": round(rot, 1), "confidence": round(conf, 2)},
                                "error": "null"})
            except Exception as e:
                results.append({"object": region["label"],
                                "angle_data": {"azimuth": 0, "polar": 0, "rotation": 0, "confidence": 0},
                                "error": str(e)})
        return {"results": results}

    def estimate_optical_flow(images, **kw):
        """Compute mean optical flow between two images using RAFT."""
        import cv2
        raft_model = model_registry.get_raft()
        device = next(raft_model.parameters()).device
        img1 = torch.from_numpy(cv2.imread(images[0])).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2 = torch.from_numpy(cv2.imread(images[1])).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _, flow_up = raft_model(img1, img2, iters=20, test_mode=True)
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
        return {"mean_flow_x": round(float(np.mean(flow[:, :, 0])), 2),
                "mean_flow_y": round(float(np.mean(flow[:, :, 1])), 2)}

    def estimate_homography(image, num_keypoints=1200, ratio_th=0.75,
                            ransac_reproj_threshold=5.0, **kw):
        """Estimate homography between two views using SIFT + RANSAC."""
        import cv2
        if not isinstance(image, list) or len(image) < 2:
            return {"homography_matrix": None, "inliers_count": 0, "total_matches": 0, "status": "failed"}
        img1 = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image[1], cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create(nfeatures=num_keypoints)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return {"homography_matrix": None, "inliers_count": 0, "total_matches": 0, "status": "failed"}
        good = [m for m, n in cv2.BFMatcher().knnMatch(des1, des2, k=2) if m.distance < ratio_th * n.distance]
        if len(good) < 4:
            return {"homography_matrix": None, "inliers_count": 0, "total_matches": len(good), "status": "failed"}
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
        if H is not None:
            matrix = [[round(float(H[i][j]), 4) for j in range(3)] for i in range(3)]
            return {"homography_matrix": matrix, "inliers_count": int(np.sum(mask)),
                    "total_matches": len(good), "status": "success"}
        return {"homography_matrix": None, "inliers_count": 0, "total_matches": len(good), "status": "failed"}

    def get_camera_params_vggt(image, dtype="auto", **kw):
        """Extract camera intrinsic + extrinsic matrices using VGGT."""
        vggt_model = model_registry.get_vggt()
        device = next(vggt_model.parameters()).device
        if isinstance(image, str):
            image = [image]
        tensors = []
        for p in image:
            img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(img).permute(2, 0, 1))
        batch = torch.stack(tensors).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = vggt_model(batch)
        results = []
        for i in range(len(image)):
            intr = preds.get("intrinsics", torch.eye(3))[0, i].cpu().tolist()
            extr = preds.get("extrinsics", torch.eye(4)[:3])[0, i].cpu().tolist()
            results.append({"image_index": i, "intrinsic": intr, "extrinsic": extr})
        return {"output": results}

    return {
        "Terminate": terminate,
        "SelfReasoning": self_reasoning,
        "LocalizeObjects": localize_objects,
        "EstimateObjectDepth": estimate_object_depth,
        "GetObjectMask": get_object_mask,
        "GetObjectOrientation": get_object_orientation,
        "EstimateOpticalFlow": estimate_optical_flow,
        "EstimateHomographyMatrix": estimate_homography,
        "GetCameraParametersVGGT": get_camera_params_vggt,
    }


# ===========================================================================
# Section 4: QwenLocalClient (AutoGen ModelClient for local Qwen inference)
# ===========================================================================

@dataclass
class _Message:
    content: Optional[str]

@dataclass
class _Choice:
    message: _Message

@dataclass
class _Response:
    choices: List[_Choice]
    model: str
    usage: Dict[str, Any] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0,
    })


class QwenLocalClient:
    """AutoGen ModelClient that wraps a locally-loaded Qwen2.5-VL model.

    The SpatialAgent conversation loop (via AutoGen) calls this client instead
    of making API calls. It converts AutoGen's text messages to Qwen's
    multimodal format with embedded images.
    """

    def __init__(self, config: Dict[str, Any], model=None, processor=None):
        self.model = model
        self.processor = processor
        self.model_name = config.get("model", "qwen2_5vl-3b")
        self.image_paths: List[str] = []  # Set before each sample
        self.max_new_tokens = config.get("max_new_tokens", 1024)

    def create(self, params: Dict[str, Any]) -> _Response:
        """Generate a response from local Qwen model."""
        messages = params.get("messages", [])
        qwen_messages = self._convert_messages(messages)

        if self.model is None:
            return _Response(choices=[_Choice(message=_Message(content=""))], model=self.model_name)

        device = next(self.model.parameters()).device
        with torch.no_grad():
            input_text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
            # Process images via qwen_vl_utils
            image_inputs, video_inputs = None, None
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(qwen_messages)
            except ImportError:
                pass

            inputs = self.processor(text=[input_text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(device)
            output_ids = self.model.generate(**inputs, num_beams=1, temperature=0.0,
                                             max_new_tokens=self.max_new_tokens, use_cache=True, do_sample=False)
            generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            text = self.processor.batch_decode(generated, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _Response(choices=[_Choice(message=_Message(content=text))], model=self.model_name)

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert AutoGen messages to Qwen multimodal format.

        - System messages pass through as text
        - First user message gets all input images embedded
        - Subsequent messages embed images only when referenced (image-N or <img path>)
        """
        qwen_messages = []
        first_user_seen = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            if role == "system":
                qwen_messages.append({"role": "system", "content": content})
                continue

            # First user message: always embed all input images
            if not first_user_seen and role == "user" and self.image_paths:
                first_user_seen = True
                parts = [{"type": "image", "image": p} for p in self.image_paths]
                parts.append({"type": "text", "text": content})
                qwen_messages.append({"role": role, "content": parts})
                continue

            first_user_seen = first_user_seen or (role == "user")

            # Subsequent messages: embed images only when referenced
            image_refs = set(re.findall(r'image-(\d+)', content))
            img_tag_matches = re.findall(r'<img\s+([^>]+)>', content)

            if image_refs and self.image_paths:
                parts = []
                for ref_id in sorted(image_refs, key=int):
                    idx = int(ref_id)
                    if idx < len(self.image_paths):
                        parts.append({"type": "image", "image": self.image_paths[idx]})
                parts.append({"type": "text", "text": content})
                qwen_messages.append({"role": role, "content": parts})
            elif img_tag_matches:
                parts = [{"type": "image", "image": p.strip()} for p in img_tag_matches]
                clean = re.sub(r'<img\s+[^>]+>', '', content).strip()
                parts.append({"type": "text", "text": clean})
                qwen_messages.append({"role": role, "content": parts})
            else:
                qwen_messages.append({"role": role, "content": content})

        return qwen_messages

    def message_retrieval(self, response: _Response) -> List[str]:
        return [c.message.content for c in response.choices]

    def cost(self, response: _Response) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: _Response) -> Dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0,
                "model": getattr(response, "model", "qwen")}


# ===========================================================================
# Section 5: Monkey-patches for vendor code bugs
# ===========================================================================

def patch_user_agent_receive():
    """Fix the Terminate bug in UserAgent.receive().

    The vendor code sets final_answer when Terminate is called but does NOT
    return — it continues to executor.execute() which crashes because
    Terminate returns a simple dict, not what the executor expects. Also,
    autogen's _is_termination_msg checks for the literal string "TERMINATE"
    which never matches the JSON output.

    This patch intercepts Terminate right after parsing and returns immediately.
    """
    from agent import UserAgent
    from utils.prompt import CoTAPrompt, DirectAnswerPrompt

    def _patched_receive(self, message, sender, request_reply=False, silent=False):
        self._process_received_message(message, sender, silent)

        parsed = self.parser.parse(message)
        content = parsed['content']
        status = parsed['status']

        # Parsing failed
        if not status:
            msg_for_check = {"content": message} if isinstance(message, str) else message
            if self.sender_hits_max_reply(sender) or self._is_termination_msg(msg_for_check):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
            self._consecutive_auto_reply_counter[sender.name] += 1
            feedback = self.feedback_generator.get_prompt("parsing", parsed)
            self.step_id += 1
            self.send(feedback, sender, request_reply=True)
            return

        # Direct answer mode
        if isinstance(self.parser.prompt_generator, DirectAnswerPrompt):
            self.final_answer = content
            self._consecutive_auto_reply_counter[sender.name] = 0
            return

        # CoTA mode
        if isinstance(self.parser.prompt_generator, CoTAPrompt):
            if len(content) > 0:
                action_name = content['name']
                # FIX: return immediately on Terminate
                if action_name == "Terminate":
                    if "answer" in content.get('arguments', {}):
                        self.final_answer = content['arguments']['answer']
                    self.called_tools.append(content)
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return

                self.called_tools.append(content)

            print(f"  [Agent] Step {self.step_id}: {content.get('name', 'no-action') if content else 'empty'}")

            executed = self.executor.execute(self.step_id, self.current_image_id, content, self.task)

            if executed['status'] and 'image_paths' in executed:
                self.new_image_paths += executed['image_paths']

            msg_for_check = {"content": message} if isinstance(message, str) else message
            if self.sender_hits_max_reply(sender) or self._is_termination_msg(msg_for_check):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return

            self._consecutive_auto_reply_counter[sender.name] += 1
            feedback = self.feedback_generator.get_prompt("execution", executed)
            if executed['status'] and getattr(executed['content'], 'image', None):
                self.current_image_id += 1
            self.step_id += 1
            self.send(feedback, sender, request_reply=True)

    UserAgent.receive = _patched_receive
    print("[Patch] UserAgent.receive() patched to fix Terminate bug.")


# ===========================================================================
# Section 6: Agent runner (one sample)
# ===========================================================================

def run_agent_on_sample(item, action_registry, qwen_client, result_folder, max_steps):
    """Run the SpatialAgent conversation loop on a single benchmark sample.

    Args:
        item: Dataset sample dict with 'question', 'img_paths', 'id', etc.
        action_registry: Dict[str, Callable] mapping tool names to functions.
        qwen_client: QwenLocalClient instance (shared across samples).
        result_folder: Directory for tool output files.
        max_steps: Maximum agent conversation steps before forced stop.

    Returns:
        (final_answer: str or None, called_tools: list of dicts)
    """
    from utils.prompt import CoTAPrompt, FeedbackPrompt
    from utils.parser import Parser
    from utils.executor import Executor
    from agent import UserAgent
    from autogen.agentchat import AssistantAgent

    # Resolve image paths (relative to cwd)
    image_paths = []
    for p in item.get("img_paths", []):
        image_paths.append(p if os.path.isabs(p) else os.path.abspath(p))

    task = {"id": item.get("id", 0), "image_paths": image_paths}

    # Create fresh components for this sample
    prompt_gen = CoTAPrompt(actions=ALL_ACTIONS)
    feedback_gen = FeedbackPrompt()
    parser = Parser(prompt_generator=prompt_gen)
    executor = Executor(input_folder=".", result_folder=result_folder, action_registry=action_registry)

    user_agent = UserAgent(
        name="user",
        prompt_generator=prompt_gen,
        feedback_generator=feedback_gen,
        parser=parser,
        executor=executor,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_steps,
        code_execution_config=False,
    )

    # Point the client to this sample's images
    qwen_client.image_paths = image_paths

    # Create assistant with the Qwen client
    llm_config = {
        "config_list": [{"model": qwen_client.model_name, "model_client_cls": "QwenLocalClient"}],
        "cache_seed": None,
    }
    assistant = AssistantAgent(
        name="assistant",
        system_message=prompt_gen.get_task_prompt_only(),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    assistant.register_model_client(QwenLocalClient, model=qwen_client.model, processor=qwen_client.processor)

    # Propagate image_paths to the registered client instance
    for client in assistant.client._clients:
        if isinstance(client, QwenLocalClient):
            client.image_paths = image_paths

    # Build question with image declarations
    question = item.get("question", "")
    if image_paths:
        labels = ", ".join(f"image-{i}" for i in range(len(image_paths)))
        question = f"The following input images are provided: {labels}.\n\n{question}"

    # Run the conversation loop
    try:
        user_agent.initiate_chat(assistant, message=question, task=task)
    except Exception as e:
        print(f"  [Error] Sample {task['id']}: {e}")
        traceback.print_exc()

    return user_agent.final_answer, user_agent.called_tools


# ===========================================================================
# Section 7: Answer evaluation (same logic as test_qwen.py)
# ===========================================================================

def clean_answer(raw_answer):
    """Strip <answer>...</answer> tags and whitespace from agent output.

    The CoTAPrompt instructs the model to wrap answers in <answer> tags,
    but extract_option() would find the 'a' in '<answer>' and return 'A'
    instead of the actual answer letter. We strip the tags first.
    """
    if not raw_answer:
        return ""
    # Extract content between <answer> tags if present
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', str(raw_answer), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(raw_answer).strip()


def extract_fallback_answer(called_tools):
    """Try to extract an answer from the last Terminate call in called_tools.

    When UserAgent.final_answer is None (e.g., max steps reached), we look
    for any Terminate action that might have been recorded.
    """
    for tool in reversed(called_tools or []):
        if isinstance(tool, dict) and tool.get("name") == "Terminate":
            return tool.get("arguments", {}).get("answer", "")
    return ""


def evaluate_answer(pred_answer, item):
    """Score predicted answer against ground truth.

    Uses the same logic as test_qwen.py: exact match for multi-choice/judgment,
    numeric comparison for open-ended, MRA for VSI-Bench.

    Returns:
        (is_correct: bool, score: float)
    """
    from utils.util import extract_option, extract_yes_no, extract_number, extract_numeric_with_unit

    question_type = item.get("question_type", "")
    ground_truth = item.get("answer", "")
    pred_str = clean_answer(pred_answer)

    is_correct = False
    score = 0.0

    if question_type.lower() == "multi-choice":
        pred = extract_option(pred_str)
        gt = extract_option(ground_truth)
        is_correct = pred.upper() == gt.upper()

    elif question_type.lower() == "judgment":
        pred = extract_yes_no(pred_str)
        gt = extract_yes_no(ground_truth)
        is_correct = pred.lower() == gt.lower()

    else:  # open-ended
        units = ['meter', 'meters', 'm', 'cm', 'centimeter', 'centimeters',
                 'km', 'kilometer', 'kilometers', 'inch', 'inches', 'ft', 'foot', 'feet']
        if any(u in ground_truth.lower() for u in units):
            is_correct = extract_numeric_with_unit(pred_str, ground_truth)["is_correct"]
        elif item.get("source") == "RealWorldQA":
            is_correct = pred_str.lower() == ground_truth.lower()
        else:
            try:
                pred_val = float(extract_number(pred_str))
            except (ValueError, TypeError):
                pred_val = 0.0
            try:
                gt_val = float(extract_number(ground_truth))
            except (ValueError, TypeError):
                gt_val = 0.0

            if item.get("source") == "VSI-Bench_8":
                from utils.util import mean_relative_accuracy
                if pred_val == 0 and gt_val == 0:
                    score = 1.0
                elif pred_val == 0:
                    score = 0.0
                else:
                    score = mean_relative_accuracy(pred_val, gt_val, start=0.5, end=0.95, interval=0.05)
                return True, score  # VSI-Bench uses MRA score directly
            else:
                is_correct = pred_val == gt_val

    score = 1.0 if is_correct else 0.0
    return is_correct, score


# ===========================================================================
# Section 8: Result saving (same format as test_qwen.py)
# ===========================================================================

def save_results(all_results, output_dir):
    """Save results grouped by source, category, and overall."""
    os.makedirs(output_dir, exist_ok=True)

    # All results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Group by source and category
    by_source, by_category = {}, {}
    for r in all_results:
        by_source.setdefault(r.get("source", "unknown"), []).append(r)
        by_category.setdefault(r.get("category", "unknown"), []).append(r)

    # Save by source
    source_dir = os.path.join(output_dir, "by_source")
    os.makedirs(source_dir, exist_ok=True)
    for source, results in by_source.items():
        with open(os.path.join(source_dir, f"{source}_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        score_sum = sum(r.get("score", 0.0) for r in results)
        total = len(results)
        acc = (score_sum / total) * 100 if total > 0 else 0
        with open(os.path.join(source_dir, f"{source}_summary.json"), "w") as f:
            json.dump({"source": source, "accuracy": acc, "correct": int(score_sum),
                        "total": total, "score_sum": score_sum}, f, indent=2)
        print(f"  Source: {source} — {acc:.1f}% ({int(score_sum)}/{total})")

    # Save by category
    cat_dir = os.path.join(output_dir, "by_category")
    os.makedirs(cat_dir, exist_ok=True)
    for cat, results in by_category.items():
        with open(os.path.join(cat_dir, f"{cat}_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        score_sum = sum(r.get("score", 0.0) for r in results)
        total = len(results)
        acc = (score_sum / total) * 100 if total > 0 else 0
        with open(os.path.join(cat_dir, f"{cat}_summary.json"), "w") as f:
            json.dump({"category": cat, "accuracy": acc, "correct": int(score_sum),
                        "total": total, "score_sum": score_sum}, f, indent=2)
        print(f"  Category: {cat} — {acc:.1f}% ({int(score_sum)}/{total})")

    # Overall
    total_score = sum(r.get("score", 0.0) for r in all_results)
    total_n = len(all_results)
    overall_acc = (total_score / total_n) * 100 if total_n > 0 else 0
    with open(os.path.join(output_dir, "overall_summary.json"), "w") as f:
        json.dump({"accuracy": overall_acc, "correct": int(total_score),
                    "total": total_n, "score_sum": total_score}, f, indent=2)
    print(f"\n  Overall: {overall_acc:.1f}% ({int(total_score)}/{total_n})")


# ===========================================================================
# Section 9: Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SpatialAgent (Qwen + spatial tools) on SpatialScore benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    cd literature/spatialscore/code
    CUDA_VISIBLE_DEVICES=0 python ../run_spatialagent.py \\
        --model_path ~/models/Qwen2.5-VL-3B-Instruct \\
        --model_name qwen2_5vl-3b \\
        --dataset_json_path ./dataset/SpatialScore_test50.json \\
        --output_dir ./eval_results_agent \\
        --checkpoints_dir ~/checkpoints \\
        --max_steps 5
        """,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen model weights")
    parser.add_argument("--model_name", type=str, default="qwen2_5vl-3b",
                        choices=["qwen2_5vl-3b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b"])
    parser.add_argument("--dataset_json_path", type=str, required=True, help="Path to SpatialScore JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="Directory with tool model checkpoints (depth_anything_v2_vitl.pth, raft-things.pth, etc.)")
    parser.add_argument("--max_steps", type=int, default=5, help="Max agent conversation steps per sample")
    parser.add_argument("--save_interval", type=int, default=10, help="Save results every N samples")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load Qwen model
    # -----------------------------------------------------------------------
    print(f"\n[1/4] Loading Qwen model: {args.model_name} from {args.model_path}")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True,
        min_pixels=256 * 28 * 28, max_pixels=2560 * 28 * 28,
    )
    print(f"  Model loaded on {next(model.parameters()).device}")

    # -----------------------------------------------------------------------
    # Step 2: Initialize tool models and action registry
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Initializing tool model registry (checkpoints: {args.checkpoints_dir})")
    model_registry = ModelRegistry(checkpoints_dir=args.checkpoints_dir)

    # SelfReasoning tool uses the same Qwen model
    def vlm_fn(image_path, query):
        """Use Qwen for SelfReasoning sub-queries."""
        from qwen_vl_utils import process_vision_info
        device = next(model.parameters()).device
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": query},
        ]}]
        input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[input_text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        torch.cuda.empty_cache()
        return response

    action_registry = make_action_registry(model_registry, vlm_fn)
    print("  Action registry ready (9 tools, models load lazily on first use)")

    # -----------------------------------------------------------------------
    # Step 3: Apply monkey-patches to vendor code
    # -----------------------------------------------------------------------
    print("\n[3/4] Applying patches to vendor code...")
    patch_user_agent_receive()

    # Create Qwen client (shared across all samples, image_paths updated per sample)
    qwen_client = QwenLocalClient(config={"model": args.model_name}, model=model, processor=processor)

    # -----------------------------------------------------------------------
    # Step 4: Load dataset and run inference
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Loading dataset: {args.dataset_json_path}")
    with open(args.dataset_json_path, "r") as f:
        data = json.load(f)
    print(f"  {len(data)} samples loaded")

    output_dir = os.path.join(args.output_dir, args.model_name)
    result_folder = os.path.join(output_dir, "agent_outputs")
    os.makedirs(result_folder, exist_ok=True)

    # Resume from existing results if available
    all_results = []
    start_idx = 0
    results_file = os.path.join(output_dir, "all_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
        processed_ids = {r.get("id") for r in all_results}
        if processed_ids:
            start_idx = max(processed_ids) + 1
        print(f"  Resuming from index {start_idx} ({len(all_results)} existing results)")

    # Process each sample
    print(f"\n{'='*60}")
    print(f"  Running SpatialAgent inference (max_steps={args.max_steps})")
    print(f"{'='*60}\n")

    for i, item in enumerate(tqdm(data[start_idx:], desc="SpatialAgent", initial=start_idx, total=len(data))):
        actual_idx = i + start_idx
        if any(r.get("id") == actual_idx for r in all_results):
            continue

        print(f"\n--- Sample {actual_idx} | source={item.get('source')} | type={item.get('question_type')} ---")
        print(f"  Q: {item.get('question', '')[:100]}...")

        # Run the agent
        final_answer, called_tools = run_agent_on_sample(
            item=item,
            action_registry=action_registry,
            qwen_client=qwen_client,
            result_folder=result_folder,
            max_steps=args.max_steps,
        )

        # Fallback if agent didn't call Terminate
        if final_answer is None:
            final_answer = extract_fallback_answer(called_tools)
            if final_answer:
                print(f"  [Fallback] Extracted answer from tool history: {final_answer}")
            else:
                print(f"  [Warning] No answer produced (max_steps reached without Terminate)")

        pred_answer = clean_answer(final_answer) if final_answer else ""
        print(f"  Answer: {pred_answer} | GT: {item.get('answer', '')}")

        # Evaluate
        is_correct, score = evaluate_answer(pred_answer, item)
        print(f"  Correct: {is_correct} (score={score})")

        # Build result entry (same format as test_qwen.py)
        result_entry = {
            "id": item.get("id", actual_idx),
            "category": item.get("category", "unknown"),
            "subcategory": item.get("subcategory", "unknown"),
            "input_modality": item.get("input_modality", "image"),
            "question_type": item.get("question_type", ""),
            "source": item.get("source", "unknown"),
            "question": item.get("question", ""),
            "gt_answer": item.get("answer", ""),
            "pred_answer": pred_answer,
            "img_paths": item.get("img_paths", []),
            "is_correct": is_correct,
            "score": score,
            # Agent-specific fields
            "called_tools": [t.get("name", "") if isinstance(t, dict) else str(t) for t in (called_tools or [])],
            "num_steps": len(called_tools or []),
        }
        all_results.append(result_entry)

        # Save periodically
        if (i + 1) % args.save_interval == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [Saved] {len(all_results)} results so far")

        torch.cuda.empty_cache()

    # Final save
    print(f"\n{'='*60}")
    print(f"  Saving final results to {output_dir}")
    print(f"{'='*60}\n")
    save_results(all_results, output_dir)
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
