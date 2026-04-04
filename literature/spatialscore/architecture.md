# SpatialAgent — Architecture Analysis

How the SpatialAgent multi-agent system works, based on the code in `code/SpatialAgent/`.

---

## 1. What is SpatialAgent?

SpatialAgent is a **training-free** wrapper around any multimodal LLM (MLLM) that improves spatial reasoning by giving the model access to **12 specialized spatial perception tools**. Instead of relying solely on the MLLM to answer spatial questions (where models struggle), SpatialAgent lets the MLLM:

1. **See** the image (perception — what MLLMs are good at)
2. **Plan** which tools to call (reasoning about strategy)
3. **Delegate** precise spatial computation to external tools (depth estimation, optical flow, orientation, etc.)
4. **Synthesize** tool outputs into a final answer

The key insight: MLLMs are good at understanding *what* is in an image but bad at computing *where* things are in 3D space. SpatialAgent externalizes the spatial computation.

**Results from the paper:**
- Qwen3-VL-8B baseline: 45.48%
- Qwen3-VL-8B + SpatialAgent (ReAct): **53.81%** (+8.33 points, no training)

---

## 2. Architecture Overview

SpatialAgent is built on a **modified AutoGen** framework using two agents in a conversation loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SpatialAgent System                      │
│                                                                 │
│  ┌──────────────┐         messages          ┌────────────────┐  │
│  │  UserAgent    │ ◄──────────────────────► │ AssistantAgent  │  │
│  │  (Orchestrator)│                          │ (MLLM Wrapper)  │  │
│  │               │                          │                  │  │
│  │  - Prompt Gen │  1. sends question       │  - System msg    │  │
│  │  - Parser     │  ──────────────────►     │  - Qwen/GPT-4V  │  │
│  │  - Executor   │                          │  - Generates     │  │
│  │  - Feedback   │  2. returns JSON with    │    {thought,     │  │
│  │               │     thought + actions    │     actions}     │  │
│  │               │  ◄──────────────────     │                  │  │
│  │               │                          │                  │  │
│  │  3. parses, executes tool                │                  │  │
│  │  4. sends observation back ──────────►   │                  │  │
│  │  5. repeat until Terminate               │                  │  │
│  └──────────────┘                           └────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │   Executor    │──► DepthAnythingV2, RAFT, OrientAnything,    │
│  │   + Tools     │    SAM2, VGGT, SIFT, ...                     │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

See `diagrams/conversation_loop.excalidraw` for an interactive version.

---

## 3. Conversation Loop (Step by Step)

Here's exactly what happens when SpatialAgent processes one question:

### Step 1: Initialize (`UserAgent.initiate_chat()`)
- Reset state: `step_id=0`, `called_tools=[]`, `final_answer=None`
- Count input images (e.g., 2 images → `image-0`, `image-1`)
- Generate initial message via `CoTAPrompt.get_prompt_for_curr_query(question)`

### Step 2: Build the Prompt (`CoTAPrompt`)
The initial message sent to the MLLM contains:
1. **Goal** — "You are a helpful assistant, solve the USER REQUEST using external tools"
2. **Action metadata** — All 9 tools with their name, description, arguments, return types, and examples
3. **Task instructions** — "Call one action at a time", "Always call Terminate at the end"
4. **Format instructions** — Output must be strict JSON: `{"thought": "...", "actions": [{"name": "...", "arguments": {...}}]}`
5. **7 few-shot demos** — Full multi-step examples showing tool usage for different spatial tasks
6. **The actual question** — e.g., "What direction is the camera moving?"

### Step 3: MLLM Generates Response (`AssistantAgent`)
The MLLM (e.g., Qwen) sees the prompt + images and generates JSON like:
```json
{
  "thought": "I need to compute optical flow between the two frames to determine camera motion direction",
  "actions": [{"name": "EstimateOpticalFlow", "arguments": {"images": ["image-0", "image-1"]}}]
}
```

### Step 4: Parse Response (`Parser.parse_cota()`)
- Extract outermost JSON from the response text
- Validate it has `thought` and `actions` fields
- Extract the first action's `name` and `arguments`
- If parsing fails → send error feedback, ask MLLM to fix format

### Step 5: Execute Tool (`Executor.execute()`)
- Look up the action function in `action_registry` dict
- Resolve image references: `"image-0"` → `/path/to/actual/image.jpg`
- Call the tool function with resolved arguments
- Wrap result in `BaseObservation` object
- If tool produces an image output, save it and increment `image_id`

### Step 6: Send Feedback (`FeedbackPrompt`)
Feedback message sent back to the MLLM:
```
OBSERVATION:
{'mean_flow_x': 2.5, 'mean_flow_y': -0.3}
The OBSERVATION can be incomplete or incorrect, so please be critical and decide 
how to make use of it. If you've gathered sufficient information to answer the 
question, call Terminate with the final answer.
```

### Step 7: Loop or Terminate
- If MLLM calls `Terminate(answer="A")` → store `final_answer`, stop
- If max steps reached (`max_consecutive_auto_reply`) → stop (answer may be None)
- Otherwise → go to Step 3 with the updated conversation history

---

## 4. Tool Registry

SpatialAgent has 9 tools (the paper mentions 12, but the released code has 9):

| Tool | Underlying Model | Input | Output | Spatial Task |
|------|-----------------|-------|--------|-------------|
| **EstimateOpticalFlow** | RAFT | 2 images | mean_flow_x, mean_flow_y | Camera/object motion direction |
| **LocalizeObjects** | Detection model (e.g., GroundingDINO) | image + object names | bounding boxes + scores | 2D object positions |
| **EstimateObjectDepth** | DepthAnythingV2 (ViT-L) | image + object names + indoor/outdoor | depth in meters per object | Distance from camera |
| **GetObjectMask** | SAM2 (Segment Anything 2) | image + object names | mask area (fraction of image) + bbox | Object size/area |
| **GetObjectOrientation** | OrientAnything (DINOv2+MLP) | image + object name | azimuth, polar, rotation angles + confidence | 3D object facing direction |
| **EstimateHomographyMatrix** | SIFT + RANSAC (OpenCV) | 2 images + params | 3x3 homography matrix | Perspective transformation |
| **GetCameraParametersVGGT** | VGGT | image(s) | intrinsic (3x3) + extrinsic (3x4) matrices | Camera pose in world coords |
| **SelfReasoning** | The MLLM itself | image + query | text response | Scene understanding fallback |
| **Terminate** | — | answer string | — | End conversation |

### How tools load
Tool models are heavy (each is a neural network). They load **lazily** — only when first called — and stay cached on GPU for subsequent calls. All tool models are open-source.

---

## 5. Key Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `SpatialAgent/agent.py` | **Orchestrator** — manages the conversation loop | `UserAgent.initiate_chat()`, `UserAgent.receive()`, `UserAgent.generate_init_message()` |
| `SpatialAgent/utils/prompt.py` | **Prompt generation** — builds system prompt with tool specs + demos | `CoTAPrompt.get_prompt_for_curr_query()`, `CoTAPrompt.get_task_prompt_only()`, `FeedbackPrompt.get_prompt()`, `DirectAnswerPrompt` |
| `SpatialAgent/utils/parser.py` | **Response parsing** — extracts JSON actions from MLLM output | `Parser.parse()`, `Parser.parse_cota()`, `extract_outermost_bracket()` |
| `SpatialAgent/utils/executor.py` | **Tool dispatch** — resolves image paths and calls tool functions | `Executor.execute()`, `Executor.get_full_image_path()` |
| `SpatialAgent/utils/observation.py` | **Result wrapper** — converts tool output dict to object with attributes | `BaseObservation.__init__()` |
| `SpatialAgent/autogen/` | **Modified AutoGen framework** — provides the multi-agent conversation infrastructure | `ConversableAgent.generate_oai_reply()`, `MultimodalConversableAgent`, `AssistantAgent`, `UserProxyAgent` |
| `SpatialAgent/DepthAnythingV2/` | Monocular metric depth estimation model | Used by `EstimateObjectDepth` tool |
| `SpatialAgent/RAFT/` | Optical flow estimation model | Used by `EstimateOpticalFlow` tool |
| `SpatialAgent/OrientAnything/` | 3D object orientation estimation | Used by `GetObjectOrientation` tool |
| `SpatialAgent/sam2/` | Segment Anything 2 | Used by `GetObjectMask` tool |
| `SpatialAgent/vggt/` | Camera parameter estimation | Used by `GetCameraParametersVGGT` tool |

---

## 6. How Spatial Reasoning is Externalized

The core idea: **split spatial QA into perception (MLLM) and computation (tools)**.

### What the MLLM does (perception + planning):
- Understands what objects are in the scene ("I see a dog and a cat")
- Reads and understands the question
- Decides which tool to call and with what arguments
- Interprets tool outputs in context ("depth 1.0m < 1.2m, so the eggs are closer")
- Formulates the final answer

### What tools do (spatial computation):
- Compute **metric depth** in meters (not just "closer/farther" but exactly how far)
- Compute **optical flow vectors** (precise pixel displacement between frames)
- Compute **3D orientation angles** (azimuth 315° = facing viewer and to the right)
- Compute **camera matrices** (intrinsic focal lengths + extrinsic pose)
- Compute **homography matrices** (geometric transformation between views)
- Compute **segmentation masks** (pixel-level object boundaries for size comparison)

### Egocentric → Allocentric transformation:
This is the key spatial reasoning challenge. Images are **egocentric** (from the camera's viewpoint), but spatial questions often require **allocentric** understanding (where things are in the world).

```
Egocentric (image)          Tools              Allocentric (world)
───────────────────    ──────────────────    ──────────────────────
Pixel coordinates   →  Depth estimation  →  3D position (X, Y, Z)
2D bounding box     →  + Camera params   →  World coordinates
Object appearance   →  Orientation est.  →  Facing direction (°)
Frame pair          →  Optical flow      →  Camera motion vector
Two views           →  Homography        →  Geometric transform
```

The MLLM can't do these transformations reliably (it's a language model, not a geometry engine). By delegating to specialized models, SpatialAgent gets precise numerical answers where the MLLM would guess.

---

## 7. Two Reasoning Paradigms

The paper describes two agent paradigms. The released code implements **CoTA (Chain-of-Thought with Actions)**, which is essentially the **ReAct** paradigm:

### ReAct / CoTA (implemented in code)
```
Think → Act → Observe → Think → Act → Observe → ... → Terminate
```
- Dynamic: each step decides the next action based on previous observations
- More flexible, better accuracy (+8.33 for 8B model)
- Slower: multiple MLLM calls per question

### Plan-Execute (described in paper, not fully in code)
```
Plan all steps → Execute all → Summarize
```
- Planner decides all tool calls upfront
- Faster: fewer MLLM calls
- Less adaptive: can't adjust based on intermediate results
- Lower accuracy (+7.27 for 8B model)

---

## 8. Known Limitations and Bugs

1. **Terminate doesn't stop the loop**: `UserAgent.receive()` sets `final_answer` on Terminate but doesn't return — continues to execute and send feedback, causing crashes. Needs monkey-patch.

2. **`<answer>` tag misparsing**: The prompt tells the model to wrap answers in `<answer>B</answer>`, but the `extract_option()` metric function finds the `a` in `<answer>` and returns `'A'` instead of the actual answer letter.

3. **No fallback answer**: When the model hits max_steps without calling Terminate, `final_answer` stays `None`, producing an empty prediction that's always wrong.

4. **Tool accuracy bottleneck**: Tool errors propagate — if object detection fails, depth estimation for that object is meaningless. The authors acknowledge the toolbox is "relatively rudimentary."

5. **Massive system prompt**: 9 tools + 7 few-shot demos create a very long system prompt. Smaller models (3B) struggle to follow the strict JSON format.
