# Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces

`CVPR 2025 Oral` · 🏛️ NYU · 🏛️ Yale · 🏛️ Stanford

[📄 Paper](https://arxiv.org/abs/2412.14171) · [💻 Code](https://github.com/vision-x-nyu/thinking-in-space) · [📊 VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) · [🚀 Project](https://vision-x-nyu.github.io/thinking-in-space.github.io/)

🏷️ **SUBJECT:** Benchmark and analysis of visual-spatial intelligence in video MLLMs.

❓ **PROBLEM:**
- No high-fidelity benchmark measures 3D spatial understanding from egocentric indoor videos in MLLMs.
- It is unclear whether MLLM failures are due to perception, language, or spatial reasoning.
- Standard prompting tricks (CoT, ToT) are assumed to help but have never been tested on spatial tasks.

💡 **IDEA:** Build a carefully annotated benchmark from 3D reconstructions (**VSI-Bench**) and use it to probe *where* MLLMs actually fail on visual-spatial intelligence — then test whether explicit **cognitive maps** can unblock them.

🛠️ **SOLUTION:**
- **VSI-Bench:** 5K+ QA pairs from 288 egocentric indoor videos (ScanNet, ScanNet++, ARKitScenes), spanning 8 tasks across configurational (direction/distance/order/route) and measurement (count/distance/size/room) categories.
- **Broad evaluation:** 15 open- and closed-source MLLMs benchmarked head-to-head against humans.
- **Linguistic analysis:** separate spatial-reasoning signal from general language capability to identify the true bottleneck.
- **Visual analysis:** elicit cognitive maps from the model and test whether explicit spatial memory improves downstream answers.

🏆 **RESULTS:** MLLMs show competitive-but-subhuman visual-spatial intelligence; **spatial reasoning — not linguistic capability — is the bottleneck**, CoT/ToT prompting actively hurts on spatial tasks, and explicit cognitive-map generation improves distance estimation.
