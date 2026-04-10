# VideoSeek — Detailed Analysis

## Architecture deep-dive

### Agent loop (agent.py)

The core loop is straightforward ReAct:

```
for step in 1..max_steps:
    1. Inject step prompt ("Step [N/20]: reason + plan next action")
    2. THOUGHT: call thinking LLM on full message history → free-text reasoning
    3. ACTION: parse thought into tool calls via a *second* LLM call with tool_choice="required"
    4. OBSERVATION: execute tool, append result to messages
    5. If action == "answer": stop
```

**Two-call design**: The thinking LLM generates free-text reasoning, then a separate LLM call with `tool_choice="required"` parses that reasoning into structured tool calls. This decouples reasoning from action parsing — the thinking model never directly sees tool schemas during reasoning.

**Message history**: The full conversation (system prompt + all prior thoughts + observations) is passed to the thinking LLM at every step. There's no summarization or memory compression — the context window grows linearly with steps.

### Toolkit implementation

| Tool | Frames sampled | Resolution | Reasoning effort | Output format |
|---|---|---|---|---|
| overview | 16α (64 default) | 256px short side, 2x4 grids | medium | JSON frame descriptions |
| skim | 4α (16 default) | 256px short side, individual | **low** | Free text with highlights |
| focus | ≤4α at ~1 FPS | Full resolution (no resize) | medium | Free text answer |
| answer | 0 (text only) | N/A | medium | Direct answer |

**Key detail**: `<overview>` packs frames into 2×4 grids (8 frames per image) to reduce the number of image tokens. `<skim>` and `<focus>` send individual frames. `<focus>` does **not** resize frames, keeping full resolution for fine-grained detail.

Wait — looking more carefully at the code: `<focus>` also does NOT resize. The `execute_focus` function in `focus.py` has no resize step, unlike `overview.py` and `skim.py` which both resize to 256px short side. This is intentional: focus needs full resolution for reading text, counting objects, etc.

### Cost structure

Each agent turn makes **2 LLM calls** minimum:
1. Thinking call (full context, reasoning)
2. Action parsing call (thought → tool call)

Plus each tool (overview/skim/focus) makes **1 additional LLM call** to process video frames. So each non-answer step = 3 LLM calls. The answer tool also makes 1 LLM call.

With an average of 4.42 turns on LVBench, that's ~13 LLM calls per question. Token usage: 49K tokens (no subs) vs GPT-5 base's 83K — fewer tokens despite more calls because far fewer frames are processed.

## Relevance to our research

### Connections to our spatial reasoning agent

VideoSeek validates a key principle we're exploring: **using VLMs for perception (what's in the frame) while externalizing reasoning (where to look, what to conclude)**. Their architecture separates:
- Perception: tool LLM calls that describe frame contents
- Reasoning: thinking LLM that plans exploration strategy

This mirrors our proposed split of using VLMs for perception while handling spatial reasoning externally.

### What they do differently from our direction

1. **No spatial reasoning**: VideoSeek's tools are purely temporal (seek to time ranges). They never reason about spatial relationships, object positions, or egocentric/allocentric transformations.

2. **Video-only, not spatial**: Their benchmarks test video understanding (what happened, when, why) not spatial reasoning (where is X relative to Y, how far, what direction).

3. **API-dependent**: Relies entirely on GPT-5 for both thinking and perception. Our approach aims to use open-weight VLMs for perception with external spatial reasoning.

### Ideas to borrow

1. **Multi-granularity toolkit pattern**: overview → skim → focus is a clean coarse-to-fine strategy. For spatial tasks, we could adapt this: `scene_overview` (identify objects and rough layout) → `spatial_scan` (estimate relationships in a region) → `measure` (precise spatial measurements).

2. **Think-act-observe with full history**: Simple but effective. The agent reasons over ALL prior observations, not just the latest. This accumulation of evidence is important for our spatial reasoning where multiple viewpoints may be needed.

3. **Decoupled thinking + action parsing**: Keeping reasoning in free text and parsing actions separately is a clean design. Our spatial agent could similarly reason in natural language about spatial relationships while using structured tool calls for measurements.

4. **Subtitle exploitation**: When subtitles are available, VideoSeek's performance jumps while frame usage drops further. This suggests that any auxiliary text signal (e.g., depth maps, scene graphs, or spatial annotations) could similarly reduce the visual burden.

### Key differences for spatial reasoning

- VideoSeek seeks evidence *in time* (temporal navigation). Our agent needs to seek evidence *in space* (different viewpoints, angles, measurement points).
- VideoSeek's observation is passive (describe what you see). Our agent needs active spatial observation (measure distances, identify reference frames, transform coordinates).
- VideoSeek uses the same model for thinking and perception. Our architecture proposes specialized external modules for spatial transformations.

## Strengths

- **Extremely efficient**: 92 frames vs 8,074 for DVD, with competitive or better accuracy
- **Clean, simple architecture**: no pre-built databases, no complex memory systems, just a loop with three tools
- **Strong empirical validation**: 4 benchmarks, comprehensive ablations, multiple case studies
- **Model-agnostic**: the agent framework works with any LMM (tested GPT-5, o4-mini, GPT-4.1)

## Weaknesses

- **Cost per question is high in LLM calls**: ~13 API calls per question at ~$0.10-0.50 each
- **No open-weight option**: entirely dependent on proprietary API (GPT-5)
- **Context window pressure**: full conversation history grows with each step; no compression or summarization
- **Limited to temporal seeking**: the toolkit can only navigate in time, not spatially within frames
- **Anomaly detection blind spot**: can't find evidence if there's no logic flow pointing to it
- **No learning**: each question starts fresh; doesn't learn from prior explorations of the same video

## Code quality notes

- Clean, well-structured codebase using `litellm` for API abstraction
- Tools are registered via a simple registry pattern
- Core data types (Action, Observation, Trajectory) are well-defined
- Config-driven with YAML files for model settings and prompts
- Uses `decord` for efficient video frame extraction
