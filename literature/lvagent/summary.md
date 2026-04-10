# LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents

`Arxiv 2024` · 🏛️ SIAT (CAS) · 🏛️ Tsinghua AIR · 🏛️ Shanghai AI Lab · 🏛️ SJTU

[📄 Paper](https://arxiv.org/abs/2503.10200) · [💻 Code](https://github.com/64327069/LVAgent)

🏷️ **SUBJECT:** Multi-agent MLLM collaboration framework for long-video question answering.

❓ **PROBLEM:**
- Feeding many frames into a single MLLM is expensive and drowns the model in redundant content.
- Single-MLLM agent pipelines bias the answer to that model's blind spots.
- Off-the-shelf CLIP retrieval has a domain gap on long videos, so relevant chunks are missed.

💡 **IDEA:** Replace the single MLLM with a **dynamically-curated team** of MLLM agents that iteratively perceive, vote, and expel the weakest member across multiple rounds of discussion.

🛠️ **SOLUTION:**
- **Selection:** rank MLLMs in an Agent Library via pseudo-label voting on 150 samples; top-3 form the team.
- **Perception:** 3-stage retrieval — random peek → generate 50-word "key info" → ASP-CLIP (finetuned on LongVR) scores 6 chunks and keeps those above 0.8.
- **Action:** each agent answers independently; if >50% agree, early-stop.
- **Reflection:** agents score each other's reasoning, expel the lowest, summarize history, loop back (up to 3 rounds).

🏆 **RESULTS:** First agent method to exceed **80% on all four long-video benchmarks** — EgoSchema 82.9, LongVideoBench 80.0 (+13.3 over GPT-4o), MLVU 83.9, VideoMME 81.7/86.6, beating 72B-scale non-agent baselines.
