# Memory Mechanisms in Video Agents — Survey (working draft v0)

- **Date:** 2026-04-23
- **Status:** v0.1 working draft. First-pass Tier 1 list verified (18 papers); top 5 cards fully expanded using the project template; 13 Tier 1 cards still at short/medium depth. Three future-dated 2026 preprints (VideoSeek, MedScope, Em-Garde) verified real.
- **Scope:** 2024–2026 papers on agentic long-video understanding with a memory mechanism (retention or gap reconstruction).
- **Brief:** [notes/brainstorming/2026-04-23-memory-survey-research-brief.md](../brainstorming/2026-04-23-memory-survey-research-brief.md)
- **Criteria:** [notes/brainstorming/2026-04-23-memory-survey-criteria.md](../brainstorming/2026-04-23-memory-survey-criteria.md)
- **Baseline reference:** LongVideoAgent (ACL 2026) — [literature/longvideoagent/](../../literature/longvideoagent/)

---

## Tier 1 — Memory mechanisms in video agents

Organized by the sub-flavor framework from the criteria: **1a** = retention
memory (store what was seen); **1b** = gap reconstruction (detect missing
info, call specialist model to fill it). A paper can be tagged with both.

### 1a — Retention memory

#### VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding

`ECCV 2024` · 🏛️ Peking University · 🏛️ BIGAI

[👤 Authors](https://arxiv.org/abs/2403.11481) · [📄 Paper](https://arxiv.org/abs/2403.11481) · [💻 Code](https://github.com/YueFan1014/VideoAgent) · [🚀 Project](http://videoagent.github.io)

🏷️ **SUBJECT:** Multimodal LLM agent for long-form video QA with a structured, tool-queryable memory.

❓ **PROBLEM:**
- VLMs alone struggle with long-horizon temporal relations in lengthy videos.
- Uniform frame sampling loses per-object continuity across minutes-long clips.
- Pure caption-based memories lose object-level state needed for tracking-type questions.

💡 **IDEA:** Split the video into a **unified memory** with two complementary stores — generic temporal event descriptions and object-centric tracking states — and let an LLM orchestrate foundation-model tools to query them zero-shot.

🛠️ **SOLUTION:**
- **Temporal memory:** segment-level event captions indexed for textual retrieval.
- **Object memory:** object-centric tracking states (identity, attributes, occurrences) from detection/tracking models.
- **Tool-using LLM:** zero-shot agent with tools for video segment localization, object memory querying, and other VLM calls.
- **Interactive loop:** the LLM plans queries against memory instead of stuffing all frames into context.

🏆 **RESULTS:** Reports average improvements of +6.6% on NExT-QA and +26.0% on EgoSchema over baselines, narrowing the gap to closed models such as Gemini 1.5 Pro.

💭 **THOUGHTS:**
- **Relevance to Su's LongVideoAgent memory-mechanism direction:** a close ancestor — the split of "event timeline" vs. "object-centric state" is exactly the kind of dual-store design Su's proposal gestures at, and object-state memory is a plausible handle for egocentric-allocentric reasoning.

---

#### VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos

`CVPR 2025` · 🏛️ UNC · 🏛️ J&J · 🏛️ KAIST

[📄 Paper](https://arxiv.org/abs/2405.19209) · 👤 Wang, Yu, Stengel-Eskin, Yoon, Cheng, Bertasius, Bansal

🏷️ **SUBJECT:** Hierarchical tree-based video memory, query-adaptive expansion.

💡 **IDEA:** Build a **query-adaptive hierarchical tree** as a structured video memory; iteratively expand branches relevant to the question ("go deeper where relevant").

🛠️ **SOLUTION highlights:**
- Tree index over temporally segmented video
- Iterative breadth/depth expansion driven by query
- LLM backbone reasons over the tree, not the raw frames
- Training-free

🏆 **EVAL:** Video-MME long split (~44 min), EgoSchema, NExT-QA.

💭 **Relevance to Su:** Tree memory is architecturally separable and would plug into LongVideoAgent as a memory structure that the frame-selection policy can query hierarchically.

---

#### MR. Video: "MapReduce" is the Principle for Long Video Understanding

`NeurIPS 2025` · 🏛️ UIUC

[📄 Paper](https://arxiv.org/abs/2504.16082) · [💻 Code](https://github.com/ziqipang/MR-Video) · 👤 Pang, Wang

🏷️ **SUBJECT:** Long-video framework with MapReduce-style cross-clip entity memory.

💡 **IDEA:** A **reduce stage** builds a shared memory of normalized characters/entities across clips; the map stage analyzes individual clips against that shared reference.

🛠️ **SOLUTION highlights:**
- Map: per-clip captioning/analysis
- Reduce: character/entity normalization into a shared memory
- Agentic long-video framework — decision layer over the memory

🏆 **EVAL:** +10% on LVBench over prior SOTA.

💭 **Relevance to Su:** Entity-normalization memory is a narrower retention design than episodic buffers; potentially a building block rather than a standalone "memory module."

---

#### Agentic Video Intelligence (AVI)

`arXiv 2025 (Nov)` · 🏛️ Southeast University · 🏛️ et al.

[📄 Paper](https://arxiv.org/abs/2511.14446) · 👤 Gao, Bao, Tu, Xu, Jin et al.

🏷️ **SUBJECT:** Training-free long-video agent with entity-graph memory and three-phase reasoning.

💡 **IDEA:** Build a **structured video knowledge base from entity graphs** as agent memory; run a Retrieve-Perceive-Review three-phase reasoning loop.

🛠️ **SOLUTION highlights:**
- Entity-graph memory built training-free from video
- Retrieve → Perceive → Review phases
- Open-source model ensemble

🏆 **EVAL:** 61.4% on LVBench, 59.8% on VideoMME-Long (hour-scale).

💭 **Relevance to Su:** Entity-graph memory is a clean target for grafting; training-free bonus. Caveat: preprint only, code status unclear.

---

#### GCAgent: Long-Video Understanding via Schematic and Narrative Episodic Memory

`arXiv 2025 (Nov)` · 🏛️ KAIST · 🏛️ ETRI

[📄 Paper](https://arxiv.org/abs/2511.12027) · 👤 Yeo, Chung, Park, Kim, Moon, Ro

🏷️ **SUBJECT:** Episodic memory structurally modeling events and causal/temporal relations.

💡 **IDEA:** Split memory into **schematic** (event structure) and **narrative** (event sequence with causal/temporal relations); a Memory Manager agent orchestrates Perception-Action-Reflection over it.

🛠️ **SOLUTION highlights:**
- Two-component episodic memory (schematic + narrative)
- Memory Manager agent with P-A-R cycle
- Event + causal/temporal structure explicit

🏆 **EVAL:** +23.5% on Video-MME Long split.

💭 **Relevance to Su:** Closest to Su's "hippocampus" framing in terminology. Code status unclear; worth reaching out to authors.

---

#### Video-EM: Event-Centric Episodic Memory for Long-Form Video Understanding

`arXiv 2025 (Aug)` · 🏛️ USTC · 🏛️ China Telecom · 🏛️ et al.

[📄 Paper](https://arxiv.org/abs/2508.09486) · 👤 Wang et al.

🏷️ **SUBJECT:** Training-free episodic-memory agent for long-form VideoQA.

💡 **IDEA:** Reframe long-form VideoQA as **episodic event construction + memory refinement**. The LLM acts as an active memory agent orchestrating off-the-shelf tools with a self-reflection loop.

🛠️ **SOLUTION highlights:**
- Keyframes → temporally ordered episodic events with spatio-temporal cues
- Self-reflection refinement step
- Training-free tool orchestration

🏆 **EVAL:** Benchmarks TBD (abstract lists claims without specifics we've verified).

💭 **Relevance to Su:** Very close conceptually to Su's framing. Preprint only; worth a deeper read in the next pass.

---

#### Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory (M3-Agent)

`Arxiv 2025` · 🏛️ ByteDance Seed · 🏛️ Zhejiang University

[👤 Authors](https://arxiv.org/abs/2508.09736) · [📄 Paper](https://arxiv.org/abs/2508.09736) · [💻 Code](https://github.com/bytedance-seed/m3-agent) · [📊 M3-Bench](https://github.com/bytedance-seed/m3-agent)

🏷️ **SUBJECT:** Real-time audio-visual multimodal agent with human-inspired episodic + semantic long-term memory.

❓ **PROBLEM:**
- Most video agents ingest visual frames only, ignoring audio/speech signals present in real streams.
- Flat caption memories conflate "what happened" with "what is known," so agents cannot accumulate stable world knowledge.
- Existing long-video QA benchmarks rarely stress cross-modal or person-centric reasoning over hours.

💡 **IDEA:** Build memory that is **entity-centric and multimodal**, separating **episodic** (what happened when) from **semantic** (durable facts) memory so an RL-trained agent can do multi-turn retrieval over accumulated experience.

🛠️ **SOLUTION:**
- **Streaming perception:** processes real-time visual and auditory inputs jointly.
- **Entity-centric memory graph:** episodic and semantic memories organized around persistent entities across modalities.
- **Memory update loop:** incrementally builds/updates memory rather than recomputing per query.
- **RL-trained agent:** multi-turn reasoning and memory retrieval, trained with reinforcement learning.
- **M3-Bench:** 100 robot-perspective + 920 web videos, QA targeting person understanding, knowledge extraction, cross-modal reasoning.

🏆 **RESULTS:** Outperforms the strongest prompting baseline (Gemini-1.5-Pro + GPT-4o) by +6.7% / +7.7% / +5.3% accuracy on M3-Bench-robot, M3-Bench-web, and VideoMME-long respectively.

💭 **THOUGHTS:**
- **Open question:** how much of the gain comes from audio vs. the episodic/semantic split — paper needs ablation inspection.
- **Relevance to Su's LongVideoAgent memory-mechanism direction:** strongest match — entity-centric episodic/semantic split is a direct template for a spatial-memory layer, and RL over retrieval trajectories is a concrete training recipe.

---

#### Memory-enhanced Retrieval Augmentation for Long Video Understanding (MemVid)

`arXiv 2025 (Mar)` · 🏛️ RUC · 🏛️ BAAI · 🏛️ Trento

[📄 Paper](https://arxiv.org/abs/2503.09149) · 👤 Yuan, Liu, Qin, Qian, Shu, Dou, Wen, Sebe

🏷️ **SUBJECT:** Cognitive-inspired four-step memory pipeline for long video QA.

💡 **IDEA:** **Four-step memory flow: memorize holistic → reason info needs → retrieve critical moments → focus**. Not self-labeled "agent" but structurally pipeline-agent-like.

🛠️ **SOLUTION highlights:**
- Holistic video memorization
- Explicit "info-need reasoning" step (closest 1a equivalent of Su's gap detection)
- Retrieval + focused attention
- Curriculum SFT + RL training

🏆 **EVAL:** MLVU, Video-MME, LVBench (all long-video).

💭 **Relevance to Su:** Clean 1a retention pipeline. The explicit "reason info needs" stage is a partial analog of Su's gap-detection trigger. Not fully agentic.

---

#### AMEGO: Active Memory from Long EGOcentric Videos

`ECCV 2024` · 🏛️ University of Bristol · 🏛️ Meta · 🏛️ Politecnico di Torino

[👤 Authors](https://arxiv.org/abs/2409.10917) · [📄 Paper](https://arxiv.org/abs/2409.10917) · [💻 Code](https://github.com/gabrielegoletto/AMEGO) · [📊 AMB](https://github.com/gabrielegoletto/AMEGO)

🏷️ **SUBJECT:** Semantic-free structured memory built from a single pass over a very-long egocentric video.

❓ **PROBLEM:**
- Egocentric streams are unstructured and too long to re-encode per query.
- Text-captioning memories are lossy for fine-grained interaction and temporal-grounding queries.
- No benchmark stressed sequencing/concurrency/temporal grounding at egocentric-video scale.

💡 **IDEA:** Mimic a human's "single-watching" retention by building a **self-contained, semantic-free representation** around **key locations and object interactions**, reusable across many queries without reprocessing frames.

🛠️ **SOLUTION:**
- **Single-pass extraction:** one traversal of the video yields a reusable memory.
- **Location tracks:** captures key scenes/locations visited during the video.
- **Object-interaction tracks:** captures hand-object interaction instances and their temporal extents.
- **Semantic-free store:** memory is index-like, queried downstream without re-running perception.
- **AMB benchmark:** 20K+ visual queries from EPIC-KITCHENS spanning sequencing, concurrency, and temporal grounding.

🏆 **RESULTS:** Surpasses prior video-QA baselines on AMB "by a substantial margin" (exact deltas [TBC from full paper]).

💭 **THOUGHTS:**
- **Open question:** how well the semantic-free representation transfers off EPIC-KITCHENS / outside kitchen-manipulation activities.
- **Relevance to Su's LongVideoAgent memory-mechanism direction:** the location + interaction schema is a compact prior for scene-grounded spatial memory and could seed a place/landmark layer in Su's agent.

---

#### Flash-VStream: Efficient Real-Time Understanding for Long Video Streams

`ICCV 2025` · 🏛️ Tsinghua · 🏛️ ByteDance

[📄 Paper](https://arxiv.org/abs/2506.23825) · [💻 Code](https://github.com/IVGSZ/Flash-VStream) · 👤 Zhang, Wang, Tang, Liu, Feng, Jin

🏷️ **SUBJECT:** Dual-memory streaming video understanding.

💡 **IDEA:** Flash Memory module with **two tiers**: low-capacity **context memory** (temporal aggregation + info-density modeling) and high-capacity **augmentation memory** (detailed spatial retrieval).

🛠️ **SOLUTION highlights:**
- Context memory: compact temporal state
- Augmentation memory: high-res retrieval store
- Real-time streaming ingestion

🏆 **EVAL:** EgoSchema, MLVU, LVBench, MVBench, Video-MME.

💭 **Relevance to Su:** Ships as an end-to-end video-LLM; the dual-memory design is conceptually portable but not demonstrated as an agent-loop component.

---

#### LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents

`ICCV 2025` · 🏛️ (affiliation TBD)

[📄 Paper](https://arxiv.org/abs/2503.10200) · [💻 Code](https://github.com/64327069/LVAgent) · 👤 Chen et al.

🏷️ **SUBJECT:** Multi-agent long-video system with Selection/Perception/Action/Reflection rounds.

💡 **IDEA:** **Dynamical multi-agent team formation** with a retrieval scheme; multi-round S-P-A-R loop distributes memory across collaborating agents.

🛠️ **SOLUTION highlights:**
- Retrieval scheme for relevant video segments
- Multi-round collaborative reasoning
- Dynamic team composition

🏆 **EVAL:** +13.3% on LongVideoBench.

💭 **Relevance to Su:** Memory here is distributed across agent messages rather than a classical bank — worth flagging as a distinct design point.

---

### 1a + 1b — Mixed retention + gap detection

#### VideoAgent: Long-form Video Understanding with LLM as Agent (Stanford)

`ECCV 2024` · 🏛️ Stanford

[📄 Paper](https://wxh1996.github.io/VideoAgent-Website/) · 👤 Wang, Zhang, Zohar, Yeung

🏷️ **SUBJECT:** LLM-as-agent long-video QA with iterative information-gap detection.

💡 **IDEA:** The agent **iteratively identifies what information is missing** and calls VLM/CLIP tools to retrieve more visual evidence. Explicit gap-detection step.

🛠️ **SOLUTION highlights:**
- Iterative "what am I missing?" reasoning
- VLM + CLIP tool calls for targeted retrieval
- Notes-in-context rather than a separate memory bank

🏆 **EVAL:** EgoSchema (long) 54.1%, NExT-QA 71.3% with ~8 frames.

💭 **Relevance to Su:** The iterative gap-detection pattern is the closest direct analog to Su's "agent detects incomplete information" step. Caveat: retention is implicit (in-context), not a dedicated memory store.

---

#### DrVideo: Document Retrieval Based Long Video Understanding

`CVPR 2025` · 🏛️ Hunan University · 🏛️ Monash

[📄 Paper](https://arxiv.org/abs/2406.12846) · 👤 Ma, Gou, Shi, Sun, Li, Rezatofighi, Cai

🏷️ **SUBJECT:** Video-as-document memory with agentic gap-detection re-querying loop.

💡 **IDEA:** Convert video into a **text-document memory**, then run an agent loop that detects missing answer-relevant info and re-queries / augments the document.

🛠️ **SOLUTION highlights:**
- Initial video → text document
- LLM agent reads document, identifies gaps
- Re-query loop augments document with targeted video evidence

🏆 **EVAL:** Video-MME long (44 min), MovieChat-1K, EgoSchema.

💭 **Relevance to Su:** Combines both flavors cleanly. The gap-detection → re-query loop is a direct pattern to study.

---

#### VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning

`arXiv 2025 (Jun)` · 🏛️ Shanghai AI Lab · 🏛️ NJU

[📄 Paper](https://arxiv.org/abs/2506.06097) · [💻 Code](https://github.com/SpXace/VideoChat-A1) · 👤 Wang, Chen, Yue, Wang, Qiao, Wang, Wang

🏷️ **SUBJECT:** Chain-of-shot reasoning as working-memory + refinement loop.

💡 **IDEA:** Iterative **shot-selection / partition / reflection** cycle; shot set evolves as working memory. Only ~7% input frames and ~12% inference time vs. baselines.

🛠️ **SOLUTION highlights:**
- Shot-selection as primary decision
- Partition to localize evidence
- Reflection step to re-select under-supported claims

🏆 **EVAL:** VideoMME 77.0 (w/subs), EgoSchema 70.1.

💭 **Relevance to Su:** Efficient frame budget is exactly Su's "60→30 frames" goal. Worth a deeper read.

---

#### VideoLucy: Deep Memory Backtracking for Long Video Understanding

`NeurIPS 2025` · 🏛️ HUST · 🏛️ NUS · 🏛️ NTU

[📄 Paper](https://arxiv.org/abs/2510.12422) · 👤 Zuo, Deng, Kong, Yang et al.

🏷️ **SUBJECT:** Hierarchical memory with progressive granularity and iterative backtracking.

💡 **IDEA:** **Hierarchical progressive-granularity memory** inspired by human recollection (coarse→fine); agent-based iterative backtracking systematically extracts information across time scales.

🛠️ **SOLUTION highlights:**
- Progressive-granularity memory tiers
- Iterative backtracking agent loop
- New benchmark: EgoMem (long temporally-unfolding events)

🏆 **EVAL:** Outperforms prior SOTA on long-video benchmarks.

💭 **Relevance to Su:** Coarse→fine progressive granularity is a natural match for Su's "efficient memory mechanisms" direction.

---

### 1b — Gap reconstruction (predominantly)

#### Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding

`NeurIPS 2025` · 🏛️ USTC · 🏛️ Microsoft Research

[👤 Authors](https://arxiv.org/abs/2505.18079) · [📄 Paper](https://arxiv.org/abs/2505.18079) · [💻 Code](https://github.com/microsoft/DeepVideoDiscovery)

🏷️ **SUBJECT:** Adaptive LLM agent that searches a multi-granular video database via tool use for hour-long video QA.

❓ **PROBLEM:**
- Long-context LLMs still degrade on information-dense hour-long videos.
- Prior video agents use fixed workflows applied uniformly across queries, wasting compute on easy ones and underperforming on hard ones.
- Single-granularity indices miss either global context or frame-level detail.

💡 **IDEA:** Frame long-video QA as **agentic search**: the LLM plans over a **multi-granular video database** (clip/segment/frame) and selects search-centric tools adaptively per query, rather than running a pre-specified pipeline.

🛠️ **SOLUTION:**
- **Multi-granular video DB:** the video is segmented and indexed at multiple granularities for retrieval.
- **Search-centric toolset:** tools tailored to browse, query, and zoom into the DB.
- **LLM planner:** reasons over observation state and chooses the next tool call, building an adaptive per-query workflow.
- **Iterative gather–reason loop:** continues until sufficient evidence for an answer is collected.

🏆 **RESULTS:** State-of-the-art on LVBench at **74.2%** accuracy (76.0% with transcripts), substantially surpassing all prior works per the authors.

💭 **THOUGHTS:**
- **Open question:** how robust the LLM planner is when the DB schema or tool surface changes — transfer across benchmarks is not quantified in the abstract.
- **Relevance to Su's LongVideoAgent memory-mechanism direction:** strong model for the "external memory + tool-use controller" split Su advocates; multi-granular indexing is a natural place to plug a spatial layer next to temporal/textual ones.

---

#### VideoExplorer: Think With Videos For Agentic Long-Video Understanding (a.k.a. VideoDeepResearch)

`Arxiv 2025` · 🏛️ Renmin University of China · 🏛️ BAAI · 🏛️ University of Trento

[👤 Authors](https://arxiv.org/abs/2506.10821) · [📄 Paper](https://arxiv.org/abs/2506.10821) · [💻 Code](https://github.com/yhy-2000/VideoDeepResearch)

🏷️ **SUBJECT:** "Thinking with video" agent that iteratively grounds sub-questions in moments and perceives at adaptive temporal scale.

❓ **PROBLEM:**
- Frame-downsampling methods lose fine-grained detail needed for hard long-video questions.
- Textual-reasoning agents over task-agnostic representations can't re-target perception to what the current sub-question needs.
- Training data for long-video reasoning trajectories is scarce.

💡 **IDEA:** Treat long-video QA as an **iterative "think-with-video" loop** that interleaves planning, temporal grounding, and task-oriented scalable perception, rather than reasoning over a static pre-computed context. *Note: the repo is named `VideoDeepResearch`; the current arXiv v6 is titled **VideoExplorer** — same project, renamed.*

🛠️ **SOLUTION:**
- **Sub-question planner:** LLM formulates the next sub-question from current state.
- **Temporal grounding:** locates relevant moments in the video for that sub-question.
- **Task-oriented scalable perception:** re-perceives grounded moments at query-appropriate temporal resolution.
- **Difficulty-adaptive training data:** long-video reasoning trajectories sampled with difficulty adaptation for high-quality hard cases.
- **Two-stage training:** supervised trajectory initialization + trajectory-level preference optimization guided by downstream rewards.

🏆 **RESULTS:** Reports "significant advantage over existing baselines" across popular long-video understanding and reasoning benchmarks; specific headline numbers [TBC from full paper — abstract does not state a single number].

💭 **THOUGHTS:**
- **Open question:** whether trajectory-level preference optimization meaningfully outperforms pure SFT on grounding/perception rewards — the abstract does not isolate this.
- **Relevance to Su's LongVideoAgent memory-mechanism direction:** the iterative "ground a sub-question, re-perceive at the right scale" loop is a concrete template for a spatial agent that pulls perception on demand rather than building a monolithic memory upfront.

---

#### VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning

`ICLR 2026` · 🏛️ NUS · 🏛️ HKUST

[📄 Paper](https://arxiv.org/abs/2503.13444) · [💻 Code](https://github.com/yeliudev/VideoMind) · 👤 Liu, Lin, Chen, Shou

🏷️ **SUBJECT:** Role-based agentic workflow with specialist-role LoRA adapters.

💡 **IDEA:** Agent decides which **specialist role** (LoRA adapter) to activate — planner, grounder, verifier, answerer.

🛠️ **SOLUTION highlights:**
- Four roles: plan / ground / verify / answer
- Each role implemented as a LoRA adapter over a shared backbone
- Role activation driven by task state

🏆 **EVAL:** 14 benchmarks including long-video VQA.

💭 **Relevance to Su:** "Memory" is implicit; mechanism is pure specialist orchestration. Useful as a design pattern for how to organize multiple specialists.

---

## Pending / borderline (Tier 1 candidates needing closer read)

- **HERMES (ICCV 2025, [arXiv 2408.17443](https://arxiv.org/abs/2408.17443))** — ECO + SeTR modules are clean episodic-compression + semantic-retrieval components with an interface any agent could call; disqualifier is lack of an agent loop in the original paper. Worth keeping as a candidate memory-module citation even if not fully Tier 1.
- **StreamAgent ([arXiv 2508.01875](https://arxiv.org/abs/2508.01875))** — anticipatory agent with hierarchical streaming KV-cache memory. Memory and agent loop both real; need to confirm long-video (>1 min) evaluation. Likely Tier 1 on closer read.
- **Adaptive Video Understanding Agent ([arXiv 2410.20252](https://arxiv.org/abs/2410.20252))** — dynamic frame sampling + feedback-driven reasoning; agentic with tool use, but no explicit memory module. Closer to Tier 2.
- **FrameMind ([arXiv 2509.24008](https://arxiv.org/abs/2509.24008))** — frame-interleaved CoT with tool-calls for targeted frames; "memory" is implicit multi-turn context. Likely Tier 1b.

## Rejected (with reasons)

- **Video-RAG (NeurIPS 2025, [arXiv 2411.13093](https://arxiv.org/abs/2411.13093))** — authors explicitly position as NOT agentic ("plug-and-play" single-turn retrieval); fails agent-loop criterion.
- **MA-LMM (CVPR 2024, [arXiv 2404.05726](https://arxiv.org/abs/2404.05726))** — memory bank is real, but architecture-level; no tool-calling. Candidate for background section.
- **LifelongMemory ([arXiv 2312.05269](https://arxiv.org/abs/2312.05269))** — first posted Dec 2023; background only.
- **MovieChat (CVPR 2024)** — short/long-term memory design but monolithic forward pass. Background only.
- **VideoChat-Flash (ICLR 2026, [arXiv 2501.00574](https://arxiv.org/abs/2501.00574))** — hierarchical compression for long-context video modeling; no agent loop.
- **LongVU ([arXiv 2410.17434](https://arxiv.org/abs/2410.17434))** — spatiotemporal token compression; no agent loop.
- **MARC ([arXiv 2510.07915](https://arxiv.org/abs/2510.07915))** — RL token compression; infrastructure-level, not agent/memory.

---

## Tier 2 — Long-video agentic (non-memory) one-liners

To be expanded in a dedicated Tier 2 pass.

- **LongVideoBench (NeurIPS 2024, [arXiv 2407.15754](https://arxiv.org/abs/2407.15754))** — benchmark, not a method.
- **VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking** — `CVPR 2026` ([arXiv 2603.20185](https://arxiv.org/abs/2603.20185), Lin et al., 2026-03-20). Think-act-observe loop with specialized seeking tools; +10.2 absolute points on LVBench with substantially fewer frames. **Verified real. Strong candidate to graduate to Tier 1b on closer read.**
- **MedScope: Think With Videos for Clinical Reasoning via Coarse-to-Fine Tool Calling** — `Arxiv 2026` ([arXiv 2602.13332](https://arxiv.org/abs/2602.13332), Li et al., 2026-02-11). Coarse-to-fine tool-calling over long clinical videos; GA-GRPO training that rewards grounded tool use; ClinVideoSuite dataset. **Verified real.**
- **Adaptive Video Understanding Agent ([arXiv 2410.20252](https://arxiv.org/abs/2410.20252))** — Amazon; dynamic sampling + reflective reasoning.
- **Select Less, Reason More ([arXiv 2510.15440](https://arxiv.org/abs/2510.15440))** — evidence-purity prioritization agent.
- **Em-Garde: Propose-Match Framework for Proactive Streaming Video Understanding** — `Arxiv 2026` ([arXiv 2603.19054](https://arxiv.org/abs/2603.19054), Zheng et al., 2026-03-19). Separates semantic understanding from streaming perception via a visual-proposal parser + lightweight matching module. **Verified real.**
- **Scaling RL to Long Videos (NeurIPS 2025)** — RL for long-video reasoning.
- **GenS: Generative Frame Sampler (ACL 2025 Findings)** — generative keyframe selection; could be a tool for a memory agent.
- **CLiViS** — embodied cognitive-map VLM reasoning over egocentric video.

---

## Synthesis (draft)

### Taxonomy of memory mechanisms observed

1. **Structured entity / graph memories** — VideoAgent-PKU (object-centric), M3-Agent (entity graph), AVI (entity KB), AMEGO (HOI + location graph).
2. **Hierarchical / tree memories** — VideoTree (query-adaptive tree), VideoLucy (progressive granularity), Flash-VStream (dual-tier).
3. **Episodic / narrative memories** — GCAgent (schematic+narrative), Video-EM (event-centric), MR. Video (cross-clip entity reduce).
4. **Document / retrieval memories** — DrVideo (video-as-document), MemVid (holistic memorize→retrieve).
5. **Specialist-tool orchestration (1b)** — Deep Video Discovery, VideoDeepResearch, VideoMind.
6. **Iterative note-taking / shot-chains** — VideoAgent-Stanford, VideoChat-A1, LVAgent.

### What could plug into LongVideoAgent

The strongest "drop-in" candidates — memory designs that could replace or augment LongVideoAgent's state without rewriting its RL-trained frame-selection policy:

| Candidate | Interface | Insertion point |
|---|---|---|
| **VideoAgent-PKU memory** | Structured object + event memory, tool-queryable | Replace in-context state with a persistent structured store |
| **M3-Agent memory graph** | Entity/face/voice nodes + episodic/semantic text nodes | Accumulates across frame-selection steps, queryable by the policy |
| **AMEGO HOI-tracklet memory** | Offline-built persistent memory | Pre-compute per-video, query at inference |
| **Deep Video Discovery DB** | Multi-granular clip DB + tool interface | Replace naive frame retrieval with granularity-aware tool set |
| **VideoDeepResearch toolkit** | Modular specialist retrievers + perceivers | Add a specialist-call layer on top of LongVideoAgent's frame-selection |

### Open gaps in the literature

- **Generative-prior gap reconstruction is rare.** Almost none of the surveyed papers implement Su's strongest-form proposal — a diffusion or video-generation model used as a prior to predict unseen frames. Most "gap reconstruction" in practice reduces to "call a retriever/detector on demand." This is a genuine white space.
- **No paper explicitly targets the "60-frame equivalent from 30 frames" compression objective** with an auxiliary generation loss, as Su sketched on 2026-04-16. This path appears unexplored.
- **Few papers evaluate on hour-scale videos.** Many claim "long video" but evaluate on sub-10-minute clips; the set genuinely evaluated on LVBench / HourVideo is smaller than the Tier 1 count suggests.
- **Memory modules and frame-selection policies are rarely jointly optimized.** Most Tier 1 work treats memory as a state the agent reads from, not as something the frame-selection policy actively shapes.
- **Hippocampus-inspired terminology is thin.** Only GCAgent and MemVid use explicitly cognitive-science-inspired framing; most others describe memory in pure engineering terms.

### Recommendations for next steps

1. ✅ ~~Fully expand the top 5 cards~~ — **done** for VideoAgent-PKU, M3-Agent, AMEGO, Deep Video Discovery, VideoExplorer / VideoDeepResearch.
2. ✅ ~~Sanity-check future-dated arXiv entries~~ — **done**. VideoSeek (CVPR 2026), MedScope, and Em-Garde all verified real.
3. **Expand the remaining 13 Tier 1 cards** to the same depth, prioritizing GCAgent and Video-EM (closest to Su's framing) and Flash-VStream (dual-memory design).
4. **Verify the pending list** (HERMES, StreamAgent, Adaptive VUA, FrameMind) with a full-paper read each.
5. **Promote VideoSeek to Tier 1b?** The CVPR 2026 abstract strongly suggests it fits — needs a closer read.
6. **Start the Tier 2 pass** once Tier 1 is fully expanded.
7. **Export to PDF** once cards are finalized.

---

## Methodology notes

- Search pass: WebSearch keyword combinations from §3.2 of the research brief, using 2024–2026 timeframe.
- Verification: every Tier 1 URL was fetched and abstract inspected. Two papers (VideoLucy, Deep Video Discovery) were independently re-verified at synthesis time — both confirmed.
- Coverage: published venues dominated the Tier 1 list (12 of 18); 6 are arXiv preprints from 2025. No pre-2024 work in primary list.
