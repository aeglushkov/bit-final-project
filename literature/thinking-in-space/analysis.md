# "Thinking in Space" — Paper Analysis & Meeting Prep

## 1. Paper Summary

**"Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces"** (Yang et al., NYU/Yale/Stanford, CVPR 2025 Oral) investigates whether video-capable MLLMs can understand 3D spatial layouts from egocentric indoor videos. The authors introduce **VSI-Bench**, a benchmark of 5,000+ spatial question-answer pairs derived from 288 indoor videos, evaluate 15 MLLMs, and find that models exhibit competitive but subhuman spatial intelligence. Crucially, they show that **spatial reasoning — not linguistic capability — is the primary bottleneck**, and that standard prompting techniques (Chain-of-Thought, Tree-of-Thoughts) actually *hurt* performance on spatial tasks. However, explicitly generating "cognitive maps" can improve spatial distance estimation.

---

## 2. What Problem Does This Paper Solve?

Humans naturally perceive spaces, remember layouts, and recall spatial information on demand. Current MLLMs can understand general videos, but **can they "think spatially"?** Specifically:

- Can they build an implicit "cognitive map" from an egocentric video walkthrough?
- Can they answer questions about object locations, distances, sizes, and routes?
- What are their failure modes — perception, reasoning, or spatial transformation?

**Why it matters:** Spatial intelligence is critical for embodied AI (robots, AR/VR, autonomous agents navigating real environments). Understanding where MLLMs fail helps guide future model development.

---

## 3. VSI-Bench: The Benchmark

### 3.1 Data Sources

| Dataset | Type | Scenes Used |
|---------|------|-------------|
| ScanNet | Indoor 3D reconstruction | Validation split |
| ScanNet++ | Higher-fidelity indoor 3D | Validation split |
| ARKitScenes | AR-captured indoor scenes | Validation split |

**Total:** 288 egocentric videos, 5,000+ QA pairs.

Videos are egocentric walkthroughs of indoor spaces (apartments, labs, factories). Ground-truth spatial information comes from 3D reconstruction annotations (object bounding boxes, room dimensions, centroids).

### 3.2 Task Types (8 tasks, 3 categories)

#### Configurational Tasks (Multiple-Choice Answer)
| Task | Question Example | What It Tests |
|------|-----------------|---------------|
| **Relative Direction** (easy/medium/hard) | "I am standing by X and facing Y, is Z to my front-left, front-right, back-left, or back-right?" | Egocentric spatial reasoning at increasing angular precision |
| **Relative Distance** | "Which of these objects is closest to the shoe rack?" | Comparative distance judgment |
| **Appearance Order** | "What is the first-time appearance order of: microwave, sofa, pillow?" | Spatiotemporal memory |
| **Route Planning** | "You are a robot beginning at the door. Navigate to the toilet. Choose: turn back, turn left, turn right, go forward." | Multi-step spatial navigation |

#### Measurement Tasks (Numerical Answer)
| Task | Question Example | What It Tests |
|------|-----------------|---------------|
| **Object Counting** | "How many chairs are in this room?" | Object detection + tracking across frames |
| **Absolute Distance** | "What is the distance between the bottle and the suitcase?" | Metric distance estimation |
| **Object Size** | "What is the length of the longest dimension of the sofa, in cm?" | Physical size estimation |
| **Room Size** | "What is the size of this room in square meters?" | Global spatial extent estimation |

#### Difficulty Breakdown for Relative Direction
- **Easy:** Left/right binary (2 options)
- **Medium:** 4 quadrants (front-left, front-right, back-left, back-right)
- **Hard:** 8 directions on a Cartesian plane (requires 135+ degree reasoning)

### 3.3 QA Generation Pipeline

```
3D Datasets (ScanNet, ScanNet++, ARKitScenes)
  → Unify into standardized meta-information format
    → Auto-generate QA pairs from 3D annotations + templates
      → Human-in-the-loop quality review (iterative)
        → Final VSI-Bench (hosted on HuggingFace)
```

Key design decisions:
- **Route Plan** is the only task with **human-annotated** QA pairs; the rest are template-generated
- Questions are filtered to remove ambiguities (e.g., two options within a threshold of each other)
- Category remapping ensures vocabulary consistency across datasets
- Small objects excluded to reduce perceptual challenges (focus on spatial, not visual acuity)

### 3.4 Evaluation Metrics

**Multiple-Choice Answer (MCA) tasks:** Accuracy (exact match)

**Numerical Answer (NA) tasks:** Mean Relative Accuracy (MRA)

MRA is the paper's novel metric. Instead of just measuring absolute error, it asks: "across a range of strictness thresholds, what fraction does the prediction satisfy?"

```
MRA = (1/S) * sum over c in [0.5, 0.55, ..., 0.95]:
         1 if |pred - target| / target <= 1 - c
         0 otherwise

Where S = number of threshold points (10 thresholds)
```

**Why MRA over MAE?** MAE doesn't distinguish between "close but wrong" and "wildly wrong" predictions. MRA provides a more discriminative, bounded [0, 1] score that captures how "approximately correct" a prediction is at varying tolerance levels.

**Chance-level baselines:**
- Chance Level (Random): random selection for MCA, not applicable for NA
- Chance Level (Frequency): always pick the most frequent answer — identifies pattern-gaming

---

## 4. Models Evaluated

### Proprietary Models (API-based)
| Model | Frames | Key Notes |
|-------|--------|-----------|
| GPT-4o (2024-08-06) | 16 | Lower frame count than others |
| Gemini-1.5 Flash | native video | Processes video natively (not frames) |
| Gemini-1.5 Pro | native video | **Best overall performer** |
| Gemini-2.0 Flash Exp | native video | Newer but not best |

### Open-Source Models
| Model | Params | Frames | Family |
|-------|--------|--------|--------|
| LLaVA-OneVision | 0.5B, 7B, 72B | 32 | LLaVA |
| LLaVA-NeXT-Video | 7B, 72B | 32 | LLaVA |
| InternVL2 | 2B, 8B, 40B | 8 | InternVL |
| VILA-1.5 | 8B, 40B | 32 | VILA |
| LongVILA | 8B | 32 | VILA |
| LongVA | 7B | 32 | LongVA |

**Evaluation setup:** Zero-shot, greedy decoding, batch_size=1, default prompts. Max 16 new tokens.

---

## 5. Key Results

### 5.1 Main Benchmark Results (Table 1)

| Model | Overall | Object Count | Abs Distance | Obj Size | Room Size | Rel Distance | Rel Direction | Route Plan | Appearance |
|-------|---------|-------------|-------------|----------|-----------|-------------|--------------|------------|------------|
| **Human Level** | **79.2** | 94.3 | 47.0 | 60.4 | 94.7 | 99.5 | 95.8 | **100.0** | — |
| Gemini-1.5 Pro | 49.1 | 90.8 | 13.8 | 96.5 | 44.8 | 99.8 | 51.2 | 99.2 | — |
| Gemini-1.5 Flash | 45.1 | 63.6 | 23.8 | 74.5 | 40.0 | 82.3 | 40.0 | — | — |
| GPT-4o | 34.9 | 46.2 | 4.5 | 41.6 | 31.6 | 71.0 | 41.3 | 31.6 | 42.6 |
| LLaVA-OneVision-72B | — | — | — | — | — | — | — | — | — |
| InternVL2-8B | 37.1 | — | — | — | — | — | — | — | — |

*(Full table in paper Table 1 — above shows representative rows)*

### 5.2 Key Findings Summary

#### Finding 1: MLLMs are competitive but significantly subhuman
- **Best model (Gemini-1.5 Pro): ~50% overall vs. ~80% human**
- The gap is **narrowest on configurational tasks** (direction, distance comparison) — models are ~94-100% on some
- The gap is **largest on measurement tasks** — absolute distance, size estimation
- Human performance on route planning: 100%. Models: varies widely

#### Finding 2: Spatial reasoning is the primary bottleneck (not language)
- Human-conducted error analysis on 163 samples from the best model
- **Error breakdown (Fig. 7):**
  - Visual perception errors: ~8%
  - Linguistic intelligence errors: ~12%
  - Relational reasoning errors: ~9%
  - **Egocentric-allocentric transformation errors: ~71%**
- This means models mostly *see* objects correctly but fail to *reason about spatial relationships* — particularly transforming between egocentric (camera-relative) and allocentric (world-relative) perspectives

#### Finding 3: Chain-of-Thought and similar prompting HURTS spatial tasks
- **This is counter-intuitive and a headline finding**
- Three prompting techniques tested: Zero-Shot CoT, Self-Consistency w/ CoT, Tree-of-Thoughts
- All three **degrade performance** on VSI-Bench on average (Fig. 8)
- Zero-Shot CoT and ToT reduce average performance by about 4%
- Self-Consistency is slightly better but still falls below baseline
- **Why?** These techniques improve *linguistic* reasoning but spatial reasoning is fundamentally different — it requires perceptual/geometric processing, not step-by-step logical decomposition
- As verification: Gemini-1.5 Pro w/ CoT gets 72.2 on VideoMME (general video QA) — a 1.6% improvement — confirming CoT helps for non-spatial video tasks

#### Finding 4: Cognitive maps improve spatial distance estimation
- Prompting the best model (Gemini-1.5 Pro) to explicitly generate a "cognitive map" (predict object center positions on a 10x10 grid) before answering questions
- **Table 3 results:**
  - Without cognitive map: relative distance accuracy = 36.0%
  - With cognitive map: relative distance accuracy = **42.0%** (+6% improvement)
  - Cognitive map accuracy correlates with answer accuracy
- Also tested on LLaVA-Video-72B and LLaVA-OneVision-72B:
  - LLaVA-Video-72B: with cog. map achieves 89% performance gain on relative distance
  - LLaVA-OneVision-72B: slight decrease (limited model capacity impairs map prediction)

#### Finding 5: Models build LOCAL, not GLOBAL spatial maps
- **Fig. 10:** Map-distance accuracy degrades dramatically with increasing object distance
- Within 1 grid unit: ~64% accuracy
- Beyond 4+ grid units: drops to ~40%
- This mirrors human cognitive map research — nearby objects are represented more accurately
- But unlike humans, models can't form a coherent *global* spatial representation

#### Finding 6: Video is essential, not just frames
- **Fig. 12 & Table 8:** Vision-enabled vs. vision-disabled comparison
  - Disabling vision causes general degradation across all tasks
  - But on object size, "Vision Disabled" models sometimes outperform (due to commonsense priors from language training)
  - Video input sequence matters: question-first then video slightly hurts Gemini performance
  - MLLMs benefit from multiple video views

---

## 6. Code Architecture

### 6.1 Repository Structure

The codebase is a **fork of the `lmms-eval` framework** (a popular LLM evaluation toolkit) with a custom `vsibench` task added.

```
thinking-in-space/
├── evaluate_all_in_one.sh          # Main entry point — configures and runs evaluation
├── lmms_eval/
│   ├── __main__.py                 # CLI entry (565 lines) — argument parsing, orchestration
│   ├── evaluator.py                # Core evaluation loop
│   ├── models/                     # 25+ model implementations
│   │   ├── llava_onevision.py      # LLaVA-OneVision
│   │   ├── internvl2.py            # InternVL2
│   │   ├── gpt4v.py                # GPT-4o (API)
│   │   ├── gemini_api.py           # Gemini (API)
│   │   ├── claude.py               # Claude (API)
│   │   └── ...
│   └── tasks/
│       └── vsibench/
│           ├── vsibench.yaml       # Task configuration
│           └── utils.py            # THE KEY FILE — all benchmark logic
├── data/meta_info/                 # 3D scene metadata (ScanNet, ScanNet++, ARKitScenes)
└── tools/                          # Dataset creation utilities
```

### 6.2 Evaluation Pipeline (How It Actually Works)

```
1. evaluate_all_in_one.sh
   - Selects model config (pretrained path, frame count, conv template)
   - Launches: accelerate launch -m lmms_eval --model X --tasks vsibench

2. lmms_eval/__main__.py
   - Parses args, loads model, initializes task
   - Downloads VSI-Bench from HuggingFace (nyu-visionx/VSI-Bench)

3. For each sample in VSI-Bench:
   a. vsibench_doc_to_visual(doc)
      → Returns video path: {dataset}/{scene_name}.mp4

   b. vsibench_doc_to_text(doc)
      → Formats: "These are frames of a video.\n{question}\n{options}\n{instruction}"
      → MCA instruction: "Answer with the option's letter from the given choices directly."
      → NA instruction: "Please answer the question using a single word or phrase."

   c. Model generates response (max 16 tokens, greedy, temp=0)

   d. vsibench_process_results(doc, results)
      → fuzzy_matching: takes first word, strips punctuation
      → MCA: exact_match(prediction, ground_truth)
      → NA: mean_relative_accuracy(float(prediction), float(ground_truth))

4. vsibench_aggregate_results(all_results)
   → Groups by question_type
   → Averages metrics per type
   → Combines direction easy/medium/hard into single "direction" score
   → Computes overall = mean of all task scores
   → Outputs table + JSON
```

### 6.3 Notable Code Details

**Prompt design is minimal:**
- Pre-prompt: "These are frames of a video."
- No spatial reasoning instructions, no hints about 3D understanding
- This is intentional — tests raw model capability without prompt engineering

**Frame sampling:**
- 32 frames uniformly sampled from video (8 for InternVL2)
- GPT-4o gets only 16 frames
- Gemini models receive native video (not frames)

**`fuzzy_matching` is very simple** (line 82):
```python
def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()
```
Just takes the first word. This means if a model says "A. front-left", only "A." is captured. This is fine for MCA but could lose nuance for NA tasks.

**MRA implementation** (lines 90-94):
```python
def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()
```
Creates ~12 threshold points from 0.5 to 0.95, checks if prediction's relative error is within each threshold, returns the mean — a smooth measure of "how close" the prediction is.

---

## 7. Critical Analysis

### 7.1 Strengths

1. **First systematic spatial intelligence benchmark for MLLMs** — fills an important gap between 2D image benchmarks and full 3D understanding
2. **Rigorous quality control** — human-in-the-loop verification with iterative refinement
3. **Novel MRA metric** — more appropriate than MAE for spatial measurement tasks
4. **Counter-intuitive finding about CoT** — "linguistic prompting hurts spatial tasks" is a significant insight that challenges the "just use CoT" default
5. **Cognitive map analysis** — connecting cognitive science concepts (cognitive maps, allocentric/egocentric frames) to MLLM evaluation is novel
6. **Comprehensive model coverage** — 15 models across proprietary and open-source families
7. **CVPR 2025 Oral** — strong venue, selected as oral presentation (top ~3% of submissions)

### 7.2 Limitations & Questions

1. **32 frames — is that enough?**
   - A video walkthrough can be minutes long; 32 frames is ~1 frame every few seconds
   - Does temporal resolution matter for spatial understanding?
   - The paper shows (Fig. 11) that # of frames has diminishing returns beyond 32

2. **Indoor scenes only — how does this generalize?**
   - All scenes are indoor rooms (apartments, labs)
   - Outdoor environments, larger buildings, or city-scale spaces are untested
   - 3D ground truth dependency limits data diversity

3. **Cognitive map experiment is narrow**
   - Only fully tested on Gemini-1.5 Pro (plus two LLaVA variants in appendix)
   - The 10x10 grid representation is coarse
   - Would more detailed spatial representations help more?

4. **No fine-tuning experiments**
   - All evaluations are zero-shot
   - Could training on spatial data improve performance?
   - This is acknowledged as future work but would strengthen the paper

5. **MRA threshold range is somewhat arbitrary**
   - Why [0.5, 0.95] specifically?
   - Different ranges would produce different relative rankings

6. **Prompt is intentionally bare-bones**
   - "These are frames of a video" — no spatial reasoning scaffolding
   - Could better prompts close the gap? The CoT finding suggests not, but there might be *spatial-specific* prompting strategies

7. **Scale bias**
   - Models might rely on commonsense priors (e.g., "rooms are typically 15-20 sq meters") rather than actual visual measurement
   - The paper partially addresses this with "Vision Disabled" experiments showing this is indeed happening for object size

### 7.3 Connection to Broader Research

| Theme | Connection |
|-------|-----------|
| **Embodied AI** | Spatial understanding is prerequisite for robot navigation; VSI-Bench tests if MLLMs could serve as "spatial reasoning modules" |
| **Cognitive Science** | Paper explicitly borrows from cognitive map theory (Tolman, O'Keefe); models show local-but-not-global mapping like some cognitive experiments |
| **3D Vision** | Bridges traditional 3D reconstruction (ScanNet) with modern MLLM capabilities |
| **Prompt Engineering** | Counter-evidence to "CoT always helps" — domain matters |
| **AR/VR** | Spatial understanding from first-person video directly applicable to AR scene understanding |
| **World Models** | Suggests MLLMs may need explicit spatial world models, not just implicit representations |

---

## 8. Discussion Questions for the Meeting

1. **"Is spatial intelligence fundamentally different from linguistic intelligence?"**
   The paper's finding that CoT hurts spatial tasks suggests yes. If so, what architectural changes would help? (separate spatial processing modules? explicit 3D representations?)

2. **"Are 32 frames enough to understand a space?"**
   Humans get continuous video. How much does temporal resolution matter? Could hierarchical frame sampling (dense at key moments, sparse otherwise) help?

3. **"Could spatial fine-tuning close the gap?"**
   The paper only tests zero-shot. What if models were fine-tuned on spatial QA data? Would that transfer across scene types?

4. **"What does the cognitive map finding mean for model architecture?"**
   If explicitly generating spatial representations helps, should future models have dedicated spatial memory modules? This connects to the "world model" debate.

5. **"How does this benchmark age?"**
   Models improve rapidly. Gemini-1.5 Pro was best at time of writing — newer models (GPT-4.5, Claude 4, Gemini 2.5) might already close the gap significantly. What would that mean for the benchmark's relevance?

6. **"What are the practical implications?"**
   If an embodied agent uses an MLLM for spatial reasoning, where would it fail? The "local but not global" map finding suggests it would navigate nearby spaces well but get lost in larger environments.

---

## 9. Quick Reference: Paper at a Glance

| | |
|---|---|
| **Title** | Thinking in Space: How MLLMs See, Remember, and Recall Spaces |
| **Authors** | Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, Saining Xie |
| **Affiliations** | NYU, Yale, Stanford |
| **Venue** | CVPR 2025 (Oral) |
| **Benchmark** | VSI-Bench: 5,000+ QA pairs, 288 videos, 8 task types |
| **Models tested** | 15 (GPT-4o, Gemini-1.5, LLaVA, InternVL2, VILA, etc.) |
| **Best model** | Gemini-1.5 Pro (~50% overall) |
| **Human performance** | ~80% overall |
| **Key insight** | Spatial reasoning (not linguistic) is the bottleneck; CoT hurts; cognitive maps help |
| **Code** | github.com/vision-x-nyu/thinking-in-space (fork of lmms-eval) |
| **Dataset** | huggingface.co/datasets/nyu-visionx/VSI-Bench |
