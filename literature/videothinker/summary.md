# VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning

`Arxiv 2026` · 🏛️ Zhejiang University · 🏛️ Fudan University · 🏛️ Wuhan University · 🏛️ Shanghai AI Lab

[📄 Paper](https://arxiv.org/abs/2601.15724)

🏷️ **SUBJECT:** Training an agentic VideoLLM that natively interleaves tool use with video perception.

❓ **PROBLEM:**
- Uniform frame sampling loses information and breaks temporal localization on long videos.
- Existing video agents reduce the VideoLLM to a passive captioner — only the LLM plans, and it can't actually *see* frames.
- Training data for multi-step video tool use requires a VideoLLM that already understands long video, which is the chicken-and-egg problem.

💡 **IDEA:** Synthesize tool-use trajectories in **caption space** with a strong agentic LLM, then **swap captions for real frames** at training time to get video-interleaved reasoning traces — bypassing the chicken-and-egg problem entirely.

🛠️ **SOLUTION:**
- **6-tool toolkit:** temporal retrieval (ClipRetrieval, SubtitleRetrieval, SubtitleSummary) and temporal zoom (FrameZoom, SubtitleZoom, CaptionZoom — the bridge tool used during synthesis).
- **Trajectory synthesis:** Qwen3-235B-MoE reasons over CG-Bench captions using the tools; 5 trajectories per sample are generated and filtered to those that match the gold answer.
- **Caption → video substitution:** CaptionZoom outputs are replaced with `<video>` tokens over actual frames, producing video-interleaved CoTs for training.
- **LoRA fine-tuning:** Qwen2.5-VL-7B trained on 10K interleaved samples, ViT frozen, max-seq-len 200K.
- **Adaptive inference:** short videos go direct; long videos retrieve top-k clips first; low answer confidence (γ < 0.7) triggers full multi-turn tool reasoning.

🏆 **RESULTS:** Qwen2.5-VL-7B + VideoThinker **matches GPT-4o on MLVU (54.8) and LVBench (48.9)** and beats the 72B base on LVBench (+1.5 pts); gains grow with video length (+3.7 on VideoMME long) and dominate all agentic LLM baselines (VideoAgent, VideoTree, VideoExplorer).
