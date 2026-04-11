# LongVideoAgent: Multi-Agent Reasoning with Long Videos

`ACL 2026 Main` · 🏛️ HKUST

[📄 Paper](https://arxiv.org/abs/2512.20618) · [💻 Code](https://github.com/longvideoagent) · [📊 LongTVQA](https://huggingface.co/datasets/longvideoagent/LongTVQA) · [📊 LongTVQA+](https://huggingface.co/datasets/longvideoagent/LongTVQA_plus) · [🚀 Project](https://longvideoagent.github.io/)

🏷️ **SUBJECT:** Tool-augmented multi-agent framework for long-form (hour-scale) video question answering.

❓ **PROBLEM:**
- Single-pass MLLMs compress or heavily downsample long video into one context window, losing the fine-grained evidence needed for questions about sparse cues.
- Prior agentic systems (e.g., VideoAgent) rely on weak, generic perception tools — insufficient for subtle object, action, and OCR cues in TV-episode footage.
- Existing frameworks underuse the LLM's planning ability and lack an RL training signal for learning *when* to invoke which tool.

💡 **IDEA:** A **multi-agent** pipeline in which a MasterAgent iteratively decides when to ground, when to look, and when to answer — coordinating a dedicated GroundingAgent (temporal localization) and VisionAgent (fine-grained perception) through a bounded action loop, and trained with GRPO on rule-based structural-plus-correctness rewards.

🛠️ **SOLUTION:**
- **MasterAgent:** policy LLM that emits exactly one structured action per turn — `<visual_query>`, `<request_grounding>`, or `<answer>` — for up to K rounds.
- **GroundingAgent:** returns a `<clip_X>` tag localizing question-relevant segments on the episode timeline (default: Grok-4-fast-reasoning, frozen).
- **VisionAgent:** reads the grounded clip and returns textual facts about objects, actions, OCR, and scene cues (default: GPT-4o, frozen).
- **AgenticRL (GRPO):** rule-based rewards combine per-step structural validity `r^fmt ∈ {0,1}` with a terminal answer-correctness reward `r^ans ∈ [0,1]`; only the master is fine-tuned.
- **LongTVQA / LongTVQA+:** new episode-level benchmarks built by aggregating all TVQA / TVQA+ clips from the same TV episode into hour-scale sequences with preserved timestamps and bounding boxes.

🏆 **RESULTS:** AgenticRL-Qwen2.5-7B reaches **60.20 / 70.80** on LongTVQA / LongTVQA+ — +14.10 / +10.50 over the non-agentic 7B baseline and on par with closed-source GPT-5-mini; Agentic-Grok tops the leaderboard at **82.65 / 85.60**, and ablations show grounding and vision agents each contribute a clean, additive gain (64.3 → 69.0 → 74.8).
