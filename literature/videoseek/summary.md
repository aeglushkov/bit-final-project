# VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

`Arxiv 2026` · 🏛️ AMD · 🏛️ University of Rochester

[📄 Paper](https://arxiv.org/abs/2603.20185) · [💻 Code](https://github.com/jylins/videoseek)

🏷️ **SUBJECT:** ReAct-style long-video QA agent that actively seeks evidence instead of densely parsing frames.

❓ **PROBLEM:**
- Existing video agents densely parse at 0.2–2 FPS, building expensive text indexes and memory stores that scale badly with video length.
- >80% of LVBench questions can be answered from <5% of frames — exhaustive parsing is wasted compute.
- Humans skim and zoom based on temporal/causal logic, but current systems have no principled way to decide *where to look next*.

💡 **IDEA:** Give a reasoning LLM a **three-granularity seeking toolkit** (overview / skim / focus) and let it run a think–act–observe loop that navigates the video's logic flow instead of consuming every frame.

🛠️ **SOLUTION:**
- **`<overview>`:** uniformly sample 16α frames across the whole video, packed into 2×4 grids, to recover the global storyline.
- **`<skim>`:** coarse scan of a candidate segment (>4α s) with 4α uniform samples at low reasoning effort.
- **`<focus>`:** dense ~1 FPS inspection of a short clip (≤4α s) with full-resolution frames.
- **Think–act–observe loop:** GPT-5 reasons over accumulated observations, plans the next tool call, and exits via `<answer>` (max 20 turns; α = 4 for LVBench, 2 elsewhere).

🏆 **RESULTS:** SOTA across long-video benchmarks while using **76–96% fewer frames** — LVBench 68.4% (76.7% w/ subs), VideoMME-long 70.1%, LongVideoBench 73.5% (29.6 frames vs. 384), beating DVD with ~1% of its frames. Swapping GPT-5 for GPT-4.1 drops accuracy to 53%, confirming reasoning strength gates tool use.

💭 **THOUGHTS:**
- **Logic-driven search has a clear blind spot:** anomaly detection and localized surprises can't be "reasoned toward" — worth thinking about when designing our own agent loop.
- **Subtitles amplify everything:** more accuracy *and* fewer frames with subs — language traces of the video's scene flow are disproportionately valuable.
