"""Action metadata definitions for SpatialAgent.

Each Action defines the metadata that CoTAPrompt uses to generate the system prompt,
and ACTION_NAMES lists the expected keys in the action_registry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Action:
    name: str
    description: str
    args_spec: Dict[str, Any]
    rets_spec: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)


TERMINATE = Action(
    name="Terminate",
    description="Terminate the conversation and return the final answer.",
    args_spec={"answer": "str - The final answer to the user's question."},
    rets_spec={},
    examples=[{"arguments": {"answer": "A"}, "returns": {}}],
)

SELF_REASONING = Action(
    name="SelfReasoning",
    description="Use the VLM's own visual reasoning to answer a sub-question about an image. Useful for scene understanding, object recognition, and general visual QA that doesn't require specialized tools.",
    args_spec={
        "image": "str - Image identifier (e.g. 'image-0').",
        "query": "str - The question to ask about the image.",
    },
    rets_spec={"response": "str - The VLM's answer to the query."},
    examples=[{
        "arguments": {"image": "image-0", "query": "Is this scene indoor or outdoor?"},
        "returns": {"response": "This scene is indoor."},
    }],
)

LOCALIZE_OBJECTS = Action(
    name="LocalizeObjects",
    description="Detect and localize objects in the image by returning bounding boxes. Uses object detection to find specified objects.",
    args_spec={
        "image": "str - Image identifier (e.g. 'image-0').",
        "objects": "list[str] - List of object names to detect.",
    },
    rets_spec={
        "regions": "list[dict] - Each dict has 'label' (str), 'bbox' ([x1, y1, x2, y2]), 'score' (float).",
    },
    examples=[{
        "arguments": {"image": "image-0", "objects": ["dog", "cat"]},
        "returns": {"regions": [
            {"label": "dog", "bbox": [120.25, 185.75, 305.85, 420.35], "score": 0.92},
            {"label": "cat", "bbox": [350.65, 210.45, 510.35, 390.20], "score": 0.88},
        ]},
    }],
)

ESTIMATE_OBJECT_DEPTH = Action(
    name="EstimateObjectDepth",
    description="Estimate the depth (distance from camera) of specified objects using monocular depth estimation. First localizes objects, then estimates their depth from the depth map.",
    args_spec={
        "image": "str - Image identifier (e.g. 'image-0').",
        "objects": "list[str] - List of object names to estimate depth for.",
        "indoor_or_outdoor": "str - 'indoor' or 'outdoor', affects depth scale.",
    },
    rets_spec={
        "results": "list[dict] - Each dict has 'object' (str), 'depth' (float in meters), 'error' (str or null).",
    },
    examples=[{
        "arguments": {"image": "image-0", "objects": ["scrambled eggs", "strawberries"], "indoor_or_outdoor": "indoor"},
        "returns": {"results": [
            {"object": "scrambled eggs", "depth": 1.0, "error": "null"},
            {"object": "strawberries", "depth": 1.2, "error": "null"},
        ]},
    }],
)

GET_OBJECT_MASK = Action(
    name="GetObjectMask",
    description="Segment objects and return their mask areas and bounding boxes. Uses object detection + SAM2 segmentation.",
    args_spec={
        "image": "str - Image identifier (e.g. 'image-0').",
        "objects": "list[str] - List of object names to segment.",
    },
    rets_spec={
        "results": "list[dict] - Each dict has 'object' (str), 'mask_area' (float, fraction of image), 'bbox' ([x1, y1, x2, y2]), 'error' (str or null).",
    },
    examples=[{
        "arguments": {"image": "image-0", "objects": ["scrambled eggs", "strawberries"]},
        "returns": {"results": [
            {"object": "scrambled eggs", "mask_area": 0.03, "bbox": [150.25, 200.75, 280.85, 340.35], "error": "null"},
            {"object": "strawberries", "mask_area": 0.0288, "bbox": [320.65, 210.45, 410.35, 330.20], "error": "null"},
        ]},
    }],
)

GET_OBJECT_ORIENTATION = Action(
    name="GetObjectOrientation",
    description="Estimate the 3D orientation (azimuth, polar, rotation) of an object in the image using OrientAnything. The azimuth angle indicates the facing direction: 0° = facing camera, 90° = facing right, 180° = facing away, 270° = facing left.",
    args_spec={
        "image": "str - Image identifier (e.g. 'image-0').",
        "objects": "str - Name of the object to estimate orientation for.",
    },
    rets_spec={
        "results": "list[dict] - Each dict has 'object' (str), 'angle_data' (dict with azimuth, polar, rotation, confidence), 'error' (str or null).",
    },
    examples=[{
        "arguments": {"image": "image-0", "objects": "person"},
        "returns": {"results": [
            {"object": "person", "angle_data": {"azimuth": 315.0, "polar": 90.0, "rotation": 0.0, "confidence": 0.89}, "error": "null"},
        ]},
    }],
)

ESTIMATE_OPTICAL_FLOW = Action(
    name="EstimateOpticalFlow",
    description="Compute average optical flow between two images using RAFT. Positive mean_flow_x indicates camera moved right, negative means left. Positive mean_flow_y indicates camera moved down, negative means up.",
    args_spec={
        "images": "list[str] - Two image identifiers (e.g. ['image-0', 'image-1']).",
    },
    rets_spec={
        "mean_flow_x": "float - Average horizontal flow.",
        "mean_flow_y": "float - Average vertical flow.",
    },
    examples=[{
        "arguments": {"images": ["image-0", "image-1"]},
        "returns": {"mean_flow_x": 2.5, "mean_flow_y": -0.3},
    }],
)

ESTIMATE_HOMOGRAPHY = Action(
    name="EstimateHomographyMatrix",
    description="Estimate the homography transformation matrix between two views of the same scene using SIFT keypoint matching and RANSAC.",
    args_spec={
        "image": "list[str] - Two image identifiers (e.g. ['image-0', 'image-1']).",
        "num_keypoints": "int - Number of SIFT keypoints to detect (default: 1200).",
        "ratio_th": "float - Lowe's ratio threshold for matching (default: 0.75).",
        "ransac_reproj_threshold": "float - RANSAC reprojection threshold (default: 5.0).",
    },
    rets_spec={
        "homography_matrix": "list[list[float]] - 3x3 homography matrix.",
        "inliers_count": "int - Number of inlier matches.",
        "total_matches": "int - Total number of matches found.",
        "status": "str - 'success' or 'failed'.",
    },
    examples=[{
        "arguments": {"image": ["image-0", "image-1"], "num_keypoints": 1200, "ratio_th": 0.75, "ransac_reproj_threshold": 5.0},
        "returns": {
            "homography_matrix": [[0.92, 0.05, -12.37], [-0.03, 0.89, 8.45], [0.0001, 0.0002, 1.0]],
            "inliers_count": 87, "total_matches": 124, "status": "success",
        },
    }],
)

GET_CAMERA_PARAMETERS = Action(
    name="GetCameraParametersVGGT",
    description="Extract camera intrinsic and extrinsic parameters from image(s) using VGGT (Vision Geometry Grounded Transformer).",
    args_spec={
        "image": "list[str] - Image identifier(s) (e.g. ['image-0']).",
        "dtype": "str - Data type ('auto', 'float32', 'float16'). Default: 'auto'.",
    },
    rets_spec={
        "output": "list[dict] - Each dict has 'image_index' (int), 'intrinsic' (3x3 matrix), 'extrinsic' (3x4 matrix).",
    },
    examples=[{
        "arguments": {"image": ["image-0"], "dtype": "auto"},
        "returns": {"output": [{"image_index": 0, "intrinsic": [[1024.3, 0.0, 512.0], [0.0, 1024.3, 384.0], [0.0, 0.0, 1.0]], "extrinsic": [[0.9986, -0.0021, 0.0523, -0.0104], [0.0045, 0.9992, -0.0398, 0.8351], [-0.0522, 0.0400, 0.9979, -0.5495]]}]},
    }],
)


ALL_ACTIONS = [
    TERMINATE,
    SELF_REASONING,
    LOCALIZE_OBJECTS,
    ESTIMATE_OBJECT_DEPTH,
    GET_OBJECT_MASK,
    GET_OBJECT_ORIENTATION,
    ESTIMATE_OPTICAL_FLOW,
    ESTIMATE_HOMOGRAPHY,
    GET_CAMERA_PARAMETERS,
]

ACTION_NAMES = [a.name for a in ALL_ACTIONS]
