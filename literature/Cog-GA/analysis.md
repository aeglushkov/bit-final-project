# "Cog-GA" — Paper Analysis

## 1. Paper Summary

**"Cog-GA: A Large Language Models-based Generative Agent for Vision-Language Navigation in Continuous Environments"** (Li, Lu, Mu, Qiao; CASIA / UCAS / HKU; arXiv 2409.02522v2, 23 Sep 2024) proposes an LLM-based generative agent for **VLN-CE** — the variant of vision-language navigation introduced by Krantz et al. (ECCV 2020) where the agent navigates continuous 3D Matterport3D scenes from a free-text instruction, with no prebuilt navigation graph. The thesis is that VLN-CE failures of LLM agents are not failures of the LLM's reasoning capacity per se but failures of **memory**, **input shaping**, and **experience accumulation**, and that all three can be externalized:

1. A **graph-based cognitive map** stores traversed waypoints (with direction + distance edges) and observed objects in a memory stream that the planner retrieves through two views — a *history chain* (already-traversed nodes) and an *observation chain* (potential targets between the previous and current position).
2. A **dual-channel ("what"/"where") scene describer** structures each panoramic waypoint into a `Go (direction), Is (room type), See (objects)` line that aligns the planner's prompt with the natural decomposition of VLN sub-instructions into "switch environment" + "find object" goals.
3. An **instruction rationalization** mechanism rewrites the active sub-instruction at every step from current observations and the unprocessed remainder of the instruction.
4. A **reflection mechanism** scores each step's deviation from ground truth and stores non-redundant reflection memory keyed by an optimal-distance + proximity + repeatability score; bottom-10% reflections are forgotten.

Cog-GA uses Vicuna-7b as the scene describer and GPT-3.5 as the high-level planner, sitting on top of the BridgingGap (Hong et al. CVPR 2022) waypoint predictor. On the VLN-CE val-unseen split (200 tasks), it reports **SR 48, OSR 59, SPL 42, NE 5.32, TL 18.3**, surpassing five prior published baselines (Waypoint, CMA, BridgingGap, LAW, Sim2Sim). Ablations show instruction rationalization is the single most load-bearing component (SR 16 without it), the cognitive map second (SR 22 without it), and reflection only marginally helpful in the short-horizon setting reported (SR 41 vs 48).

---

## 2. What Problem Does This Paper Solve?

VLN-CE was introduced precisely because classical VLN (Anderson et al. 2018, R2R) overstates an agent's spatial competence by giving it a discrete navigation graph. In VLN-CE the agent must instead pick a **continuous low-level action** (turning angle 3° increment over 120 angles, forward step 0.25 m increment over 12 distances → 0.25–3.00 m) at each step from a 360° panoramic RGB-D observation. This exposes three concrete failure modes for an LLM agent:

1. **The waypoint search space is large and noisy.** A waypoint predictor produces a heatmap over 120×12 = 1440 cells; even after non-max suppression there are too many candidate waypoints for an LLM to evaluate efficiently. The authors retain Hong et al.'s waypoint predictor and limit candidates to top-7.
2. **LLMs lack persistent spatial memory.** The system prompt grows linearly with trajectory length and the LLM cannot natively track "I came from the living room four steps ago", let alone backtrack. Without external memory, LLM-as-planner agents either repeat routes or oscillate.
3. **Raw natural-language instructions are too long.** A typical R2R-CE instruction has 4–6 sub-targets; presenting all of them at once dilutes the planner's attention. Naive sub-instruction splitting also fails because individual sub-instructions are ambiguous out of context — "Exit the living room" doesn't tell the LLM what to look for next.

The paper claims a single coherent design — graph cognitive map + dual-channel scene description + instruction rationalization + reflection — fixes all three. Section IV-D's ablations support that ordering: the most damaging component to remove is instruction rationalization (SR 48 → 16), which validates failure mode 3 as the dominant blocker.

---

## 3. Method

### 3.1 System Architecture

```
Panorama (RGB) ──────────────────────────────┐
                                              │
Instruction ──► Instruction Processor ────┐   │
                (rationalization Eq. 2)   │   │
                                          ▼   ▼
Waypoint Predictor (heatmap 120×12)  ──► Scene Describer (Vicuna-7b)
                                       │   "what" stream: landmarks
                                       │   "where" stream: room type / direction
                                       ▼
                                   structured prompt
                                "Go (direction), Is (room type), See (objects)"
                                       │
                              ┌────────▼───────────┐
                              │ High-Level Planner │ ◄── Memory Stream
                              │     (GPT-3.5)      │       ├── Cognitive Map G(E,N)
                              │                    │       │     ├── history chain
                              │                    │       │     └── observation chain
                              │                    │       └── Reflection Memory
                              └────────┬───────────┘
                                       ▼
                              Target Waypoint Index
                                       │
                                       ▼
                        Low-Level Actuator (angle + distance)
                                       │
                                       ▼
                              Reflection Generator
                              (DTW vs ground-truth path) ──► Memory Stream update
```

The flow per step (Algorithm 1 in the paper):

1. Waypoint predictor produces candidate waypoints `W = {o_1, …, o_m}` from the panorama.
2. Scene describer computes `(D_k, r_k)` — the dual-channel description and room-type tag — for each candidate.
3. Planner picks `target = Planner(I_{i,j}, G, M, {D_k, r_k})` — uses current rationalized sub-instruction, cognitive map, memory stream, and waypoint descriptions.
4. Actuator executes `y = T(target)`.
5. Update path P(y), update cognitive map G(E,N), generate reflection `exp = Reflection(y, y*, G, M, I_{i,j})`, update memory.
6. Rationalize next sub-instruction `I_{i,j+1} = R(I_{i,j} | D, k=1..m, I)`.
7. If sub-instruction `i` is complete, advance `i ← i+1`. If full instruction is complete, halt.

### 3.2 Cognitive Map G(E, N)

The map is an undirected graph

```
G({E_{p,o}, E_p}, {N_p, N_o})
```

with two node types and two edge types:

| Symbol | Description |
|---|---|
| `N_p` | traversed waypoint nodes; each carries a time-step label `t` |
| `N_o` | observed object nodes (one per object detection) |
| `E_p` | edges between waypoint nodes — weighted by **distance** (0.25–3 m) and **direction** (1-of-8 cardinal, encoded 1–8) |
| `E_{p,o}` | weight-1 edges connecting an object node to the waypoint where it was seen |

Two retrieval methods are defined (Figure 3):

- **History chain (B):** the chronological path of `N_p` nodes already navigated. Used by the planner as an "abstract view of the current path".
- **Observation chain (C):** the set of `N_p` nodes between the agent's *current* position and the *previous* position, along with their attached `N_o` objects. Used as a "broader view of past decisions" — what was visible from where, which is what the planner needs to backtrack or to confirm a missed target.

**Notable design choices:**

- The map is a *symbolic* graph the LLM reads as text, not a metric occupancy map. There is no ego↔allo coordinate transform module — direction is an integer 1–8.
- Object nodes are cheap (one per detection per waypoint) so the graph grows fast; the paper does not report typical graph sizes per task or memory caps.
- The graph is purely *online* — no map is shared across episodes. Reflection memory is the only cross-episode persistence.

### 3.3 Dual-Channel Scene Description

Each candidate waypoint is described in two streams:

- **What stream** — landmark objects (e.g. `['sofa', 'bookshelf', 'window']`)
- **Where stream** — room type + spatial characteristics (e.g. `In: living room`, direction `go Left Front`)

The describer outputs strings like

```
Go (Left Front for 1.25 m), Is (living room), See (sofa, picture frame, lamp, bookshelf, window, clock, …)
```

(Step samples in Appendix Section 5 confirm this exact template.)

The motivation: VLN sub-instructions naturally factor as `(switch environment, find target)` — the *where* stream supports environment-switching sub-goals, the *what* stream supports object-finding sub-goals, and pairing them aligns prompt structure with task structure.

The structural guidance lines `"You should try to go (where)"` and `"You should try to find (what)"` are appended to the rationalized sub-instruction so the LLM knows which channel is currently load-bearing. (Appendix Section 2 calls this an "Optimal Prompt Mechanism" found through prompt engineering.)

### 3.4 Instruction Rationalization

The instruction processor splits the original instruction `I` into a sub-instruction set `{I_{1,0}, I_{2,0}, …, I_{n,0}}`. At each time step, the *current* sub-instruction is rewritten:

```
I_{i, j+1} = R(I_{i, j} | D, k=1..m, I)             (Eq. 2)
```

i.e., the new version of sub-instruction `i` is conditioned on the most recent waypoint descriptions `D_{k, k=1..m}` and the unprocessed full instruction `I`. The example in the paper:

| Step type | Text |
|---|---|
| Original sub-instruction | "Exit the living room." |
| Rationalized sub-instruction | "Find the door of the living room and look for the sign to the kitchen." |

This is the most consequential single design choice — its ablation alone costs SR 32 points (48 → 16). The mechanism is essentially online prompt rewriting that grounds an abstract sub-target into a concrete perceptual cue available *now*.

### 3.5 Reflection Mechanism

After every step the Reflection Generator computes:

- `d_m` — DTW distance between current and ground-truth navigation sequences (the "optimal distance")
- `t_m` — proximity, the time-closeness to the current step
- `r_m` — repeatability, frequency of similar memories

Score:

```
Score_m = |d_m − δ| / δ  +  t_m / T  +  r_m / max(r_n)            (Eq. 3)
```

with `δ` a threshold parameter, `T` the current time step, `R` the set of repeatabilities. Identical new memories are not stored — instead the existing memory's proximity/repeatability is updated. The bottom-10% of reflection memories are forgotten.

The reflection memory carries a textual lesson (Figure 5 example: *"Move towards the exit of the living room and look for any signs of an exit"*). At inference the planner is presented with the top-scored reflection memories alongside the cognitive map.

**This is the part of the system that requires a ground-truth path.** That's a non-trivial assumption the paper does not flag clearly: it's compatible with replaying training episodes for offline experience accumulation, but the abstract's framing ("captures feedback from prior navigation experiences, facilitating continual learning and adaptive replanning") oversells it for the deployment setting.

### 3.6 Concrete Models and Hyperparameters

- **Scene describer:** Vicuna-7b. The paper says it "aligns visual modality information with natural language information" but does *not* specify a vision encoder, ViT-LLaVA bridge, or any image-to-token mechanism. The likely setup is BLIP/LLaVA-style features piped to Vicuna, but this is left implicit.
- **Planner:** GPT-3.5 (OpenAI API).
- **Waypoint predictor:** from BridgingGap [10] (Hong et al. CVPR 2022), candidates capped to 7.
- **Action space:** 120 angles × 12 distances = 1440 micro-actions; 3° turning increments and 0.25 m forward step.
- **Hardware:** 2× NVIDIA RTX 4090.
- **Eval set size:** 200 tasks (limited by LLaMA latency — the paper acknowledges this).

---

## 4. Experiments

### 4.1 Comparison with Prior VLN-CE Methods (Table I)

VLN-CE val-unseen, 200 tasks, Matterport3D.

| Method | NE ↓ | TL | SR ↑ | OSR ↑ | SPL ↑ |
|---|---:|---:|---:|---:|---:|
| Waypoint [15] | 6.31 | 7.62 | 36 | 40 | 34 |
| CMA [17] | 7.60 | 8.27 | 29 | 36 | 27 |
| BridgingGap [10] | 5.74 | 12.2 | 44 | 53 | 39 |
| LAW [21] | 6.83 | 8.89 | 35 | 44 | 31 |
| Sim2Sim [16] | 6.07 | 10.7 | 43 | 52 | 36 |
| **Cog-GA (Ours)** | **5.32** | 18.3 | **48** | **59** | **42** |

Two things to notice:

1. **OSR – SPL gap of 17** (59 − 42) is by far the largest in the table. OSR rewards getting *near* the goal at any point on the trajectory; SPL penalizes path length and rewards *correct stopping*. Cog-GA navigates well but stops conservatively — exactly what the paper acknowledges in Section IV-C ("the agent's conservative stopping mechanism, which prefers to get as close to the target point as possible").
2. **TL = 18.3 is 1.5–2.4× longer** than every baseline. The cognitive-map + reflection design buys success rate, but at a real efficiency cost. SPL (which discounts by path length) shows only a +3 lift over BridgingGap (39 → 42).

### 4.2 Ablations (Table II)

200 tasks, val-unseen.

| Variant | SR ↑ | OSR ↑ | SPL ↑ |
|---|---:|---:|---:|
| (−) Reflection | 41 | 57 | 38 |
| (−) Rationalization | 16 | 33 | 24 |
| (−) Cognitive Map | 22 | 46 | 32 |
| **Cog-GA (full)** | **48** | **59** | **42** |

**Component importance (∆SR vs full):**

- Instruction rationalization: −32
- Cognitive map: −26
- Reflection: −7

The first two losses are catastrophic (SR < 25). The reflection loss is small. The paper's gloss — "the reflection mechanism is primarily used for experience accumulation, suggesting that its importance will grow over the long term as more reflective memory is accumulated" — is plausible but unsubstantiated. There is no long-horizon ablation to validate this claim.

### 4.3 Qualitative Trace (Appendix Section 5)

The 20-step worked example ("Exit the living room and turn right into the kitchen…") gives a clean view of how the dual-channel descriptions evolve and how the agent's stopping behavior plays out:

- Distance to goal trajectory: 14.4 → 13.6 → 13.7 → 12.4 → 12.1 → 11.5 → 12.2 → 10.1 → 8.9 → 6.9 → 8.3 → 8.1 → 6.4 → 5.8 → 7.5 → 5.3 → 3.5 → 5.5 → 3.2 → **2.1 (stop)**.
- The agent enters the kitchen at step 6 and oscillates between kitchen and living room (steps 7–19) before stopping.
- This is a textbook example of the **OSR ≫ SPL** behavior — the agent reaches OSR at step 17 (3.5 m, just barely outside the 3 m threshold) but doesn't commit to stopping until step 20.

The trace is consistent with the paper's claim that the cognitive map enables backtracking, but it also illustrates the cost: the agent visits the kitchen, leaves, returns, leaves, and returns again before stopping.

---

## 5. Camera / Sensor Assumptions

VLN-CE provides:

- **360° panoramic RGB-D** observation per step (depth is part of the standard VLN-CE setup).
- **Discrete agent pose** (x, y, heading) — the simulator (Habitat) maintains this internally.
- **No external maps** — that's the point of VLN-CE.

The cognitive map's "direction" edges (1–8 cardinal) and "distance" edges (0.25–3 m) are computable directly from Habitat's pose stream. No SLAM, no DUSt3R-like estimation, no monocular pose recovery. This is a much cleaner sensor setup than Embodied VideoAgent's — and means **Cog-GA's robustness to noisy poses is untested**.

For the reflection mechanism, the system additionally needs the **ground-truth navigation path** during training/evaluation to compute DTW. The paper does not discuss what happens if reflection is built up from agent's own trajectories (no GT) — a real-deployment question.

---

## 6. Critical Analysis

### 6.1 Strengths

1. **Concrete instantiation of the agent-on-top-of-LLM pattern for VLN-CE.** Most prior VLN-CE work is end-to-end (RL or imitation learning over CMA-style cross-modal attention). Cog-GA is one of the first generative-agent approaches to actually beat those baselines on VLN-CE.
2. **Dual-channel ("what"/"where") prompt template is generic and simple.** The `Go (direction), Is (room type), See (objects)` line is a textual scaffold that can be reused for any panoramic-VLN setting; it is also legible enough that ablation of either channel is straightforward.
3. **Honest ablation reveals which components actually matter.** The −16/−22/−41 ladder is unusual to publish — most papers hide the fact that the headline component (here, reflection) is not the load-bearing one.
4. **Beats five published baselines on the canonical VLN-CE benchmark** with Vicuna-7b + GPT-3.5, no fine-tuning, no scene-specific training. The cost is paid in API calls and trajectory length, not model training.
5. **Cognitive map is symbolic and inspectable.** Direction is one of 8 cardinals, distance is a real number, time-step is an integer. Easy to debug, easy to ablate, easy to extend.

### 6.2 Limitations and Open Questions

1. **Reflection requires ground-truth path.** The reflection score uses DTW against the correct sequence (`d_m` in Eq. 3). Without GT, the reflection mechanism doesn't have its primary signal — the paper does not analyze this dependency. For a robot in a novel environment, only steps 1–4 of the algorithm survive; reflection is a training-time extra.
2. **SPL gain over BridgingGap is small (+3).** Cog-GA's headline claim is a +4 SR lift over BridgingGap. With trajectory length 1.5× longer, the *path-efficient* metric (SPL) is barely better. The qualitative agent that "successfully navigates 48% of episodes after exhaustive backtracking" is not the same as one that *plans correctly the first time*.
3. **Vicuna-7b "scene describer" lacks vision encoder details.** The paper repeatedly calls Vicuna-7b the visual-text aligner without specifying a vision module. In practice this is probably LLaVA-style features. Reproducibility suffers; comparisons to other scene-description backbones are impossible.
4. **No comparison to LLM-agent baselines on VLN-CE.** Compared to Waypoint, CMA, BridgingGap, LAW, Sim2Sim — all non-LLM. There's no head-to-head with another LLM-agent VLN system (e.g. NavGPT, A2Nav, VELMA in the related work). So we know "Cog-GA > non-LLM VLN-CE methods" but not "Cog-GA > best LLM-agent VLN-CE method".
5. **200-task eval is small.** Standard VLN-CE val-unseen has ~1839 episodes. Restricting to 200 (LLaMA latency) means the reported numbers have unmeasured variance and ablation deltas may be noisy.
6. **Reflection's "long-term benefit" is asserted but not measured.** Section IV-D says reflection's importance "will grow over the long term as more reflective memory is accumulated." There is no experiment varying memory budget, episode count, or transfer to harder splits to actually show this.
7. **No discussion of conflict between cognitive map and reflection.** Reflection memory is text the planner reads alongside the graph. If the two disagree (e.g. graph says "kitchen is north" but reflection says "I went south last time and ended up in the kitchen"), there is no arbitration logic.
8. **Conservative stopping is a known issue but not fixed.** The paper acknowledges TL is high because of conservative stopping. A reasonable fix (a stopping head, or a "I am at the goal" classifier) is not attempted.
9. **No code release at the time of this analysis.** Only the paper PDF is in `literature/Cog-GA/`. None of the implementation specifics (LangChain prompts, Habitat config) are reproducible from the paper alone.
10. **Categorization of "generative agent" is loose.** The Generative Agents (Park et al.) framing leans on the reflection + memory motif; Cog-GA inherits the vocabulary but its reflection is much more constrained (DTW score) and its memory is task-specific. The "generative agent" branding is partly aspirational.

### 6.3 Mapping to Project Research Hypothesis

The project's thesis: **VLMs excel at perception and fail at spatial reasoning; an external agent layer can supply the spatial reasoning.** Where does Cog-GA fit?

| Aspect | Inside the LLM (GPT-3.5) | Outside the LLM |
|---|---|---|
| Object detection | | ✓ (waypoint predictor + Vicuna scene describer) |
| Direction encoding | reads 1–8 integer | ✓ derived from Habitat pose |
| Distance encoding | reads meters | ✓ derived from Habitat pose |
| Cognitive map (graph) | reads as text | ✓ maintained by agent code |
| History chain / observation chain retrieval | | ✓ graph traversal |
| Sub-instruction rationalization | ✓ (LLM rewrites) | scaffolded by current observation |
| Waypoint scoring | ✓ (LLM picks index) | candidates filtered to top-7 |
| Stopping decision | ✓ (LLM) | |
| Reflection scoring | | ✓ DTW + closed-form Eq. 3 |
| Reflection content | ✓ (LLM writes the text lesson) | |

Compared to **Embodied VideoAgent**:

- Both externalize **memory** to a graph-like structure.
- Embodied VideoAgent externalizes **3D geometry** (bboxes, IoU, containment) to deterministic code; Cog-GA does not — its "spatial reasoning" lives in graph traversal that the LLM does by reading text. There is no explicit ego↔allo transform anywhere in Cog-GA; it is all relative/topological.
- Embodied VideoAgent's VLM is asked perceptual questions ("is this the target?"); Cog-GA's LLM is asked planning questions ("which waypoint?"). The latter is closer to what TiS argues *fails* in pure-VLM benchmarks.

**Net for our direction:** Cog-GA validates the *architecture pattern* (cognitive map + scaffolded prompts + reflection) but does not validate that *spatial reasoning is externalized* — the LLM is still the spatial reasoner, just one with a richer text representation in its prompt. The paper sits one notch closer to "agent helps VLM" than to "agent replaces VLM's spatial reasoning."

The papers most directly on this same axis from `literature/`:

- **Embodied VideoAgent**: same agent-loop philosophy, much more aggressive externalization of geometry; complementary.
- **Thinking-in-Space**: VSI-Bench evaluates VLM spatial reasoning in egocentric video QA — Cog-GA doesn't address QA but its memory abstraction is exactly the kind of scaffolding TiS argues for.
- **SAVVY / 3D-Mem / Feature4X / LIRA**: 3D-scene-representation siblings; these have *richer* spatial structure than Cog-GA's 1–8 cardinal graph and would be obvious upgrades.
- **3D-Mem, EmbodiedVideoAgent**: object-centric memory; Cog-GA is waypoint-centric. Different granularities — combining them is an interesting open direction.
- **VLN-Bridge / BridgingGap**: Cog-GA literally builds on top of BridgingGap's waypoint predictor (top-7 candidates). The improvement is purely in agent + memory, not in the perceptual front-end.

---

## 7. Concrete Ideas Sparked

1. **Cognitive map for VSI-Bench-style QA.** Cog-GA's symbolic graph (waypoints + 1–8 directions + distances) is exactly the structure that TiS's "egocentric → allocentric" failure points to as missing. Try inserting a Cog-GA-style cognitive map (built from frame-level pose + monocular distance estimates) as a tool the QA agent can query, and see how much of the VSI gap closes.

2. **Replace 1–8 cardinal direction with metric coordinates.** Cog-GA's directional encoding is 8-bin and the LLM does the "left-vs-right-rear" reasoning textually. A small change: store metric (x, y) and let the agent code do the relational predicate ("is target to the left of current heading?"). Compare SR/SPL on val-unseen.

3. **Decouple reflection from ground truth.** The reflection score depends on DTW vs ground-truth path. Replace `d_m` with a *self-consistency* score (e.g. agreement between cognitive-map-predicted goal location and observed goal location at task end). This removes the implicit GT dependency and turns reflection into a true self-supervised mechanism.

4. **Stopping head as a separate module.** OSR − SPL = 17 means the agent could be 17 SPL points better with a perfect stopping classifier. Try adding a stopping head that consumes the same prompt + the *what*/*where* description and outputs Bernoulli stop. This is the lowest-hanging fruit for Cog-GA-class systems.

5. **Reflection long-horizon study.** The paper claims reflection's value compounds over time but does not show it. Run Cog-GA on the full ~1839 val-unseen with reflection memory **persistent** vs **reset per episode** and report SR/SPL deltas as a function of episode index. This is exactly the experiment missing from Section IV-D.

6. **Compare Cog-GA's instruction rationalization to step-conditioned chain-of-thought.** Eq. 2's online rewriting is a special case of CoT prompting. An ablation comparing (a) static sub-instructions, (b) Cog-GA rationalization, (c) free-form CoT scratchpad would pinpoint whether the gain is from *rewriting* or from *grounding rewrites in current observation*.

7. **Combine with Embodied VideoAgent's persistent object memory.** Cog-GA's `N_o` nodes carry only category labels. Adding 3D bboxes + visual features (à la EVA's `M_O`) would let the planner answer object-state questions ("is the door I saw earlier still open?") and would close the gap to QA-style spatial reasoning.

---

## 8. Quick Reference

| | |
|---|---|
| **Title** | Cog-GA: A Large Language Models-based Generative Agent for Vision-Language Navigation in Continuous Environments |
| **Authors** | Zhiyuan Li, Yanfeng Lu, Yao Mu, Hong Qiao |
| **Affiliation** | CASIA · UCAS · University of Hong Kong |
| **Venue** | arXiv 2409.02522v2 (23 Sep 2024) |
| **Builds on** | BridgingGap [Hong et al. CVPR 2022] (waypoint predictor); Generative Agents [Park et al.] (memory + reflection motif) |
| **Memory** | Cognitive Map G(E,N) with traversed-waypoint nodes (8-cardinal directions, 0.25–3 m distances, time-step labels) and observed-object nodes (1-weight edges) + Reflection Memory (text lessons scored by DTW + proximity + repeatability) |
| **Perception stack** | BridgingGap waypoint predictor (heatmap 120 angles × 12 distances) → Vicuna-7b scene describer (dual-channel "what"/"where") |
| **Planner** | GPT-3.5 in a generative-agent loop (Algorithm 1); top-7 waypoint candidates per step |
| **Sensor inputs** | 360° panoramic RGB-D + Habitat pose; ground-truth path used by reflection generator |
| **Benchmark** | VLN-CE val-unseen (Matterport3D, 200 tasks); metrics NE / TL / SR / OSR / SPL |
| **Headline numbers** | NE **5.32** · TL 18.3 · SR **48** · OSR **59** · SPL **42** (best prior BridgingGap: 5.74 / 12.2 / 44 / 53 / 39) |
| **Ablations (∆SR)** | −Rationalization −32 · −CognitiveMap −26 · −Reflection −7 |
| **Hardware** | 2× NVIDIA RTX 4090 |
| **Code** | not released at time of this analysis (only PDF in `literature/Cog-GA/`) |
