# Research brief: survey of memory mechanisms in video agents

- **Date:** 2026-04-23
- **Who this is for:** A researcher (or student) doing the literature search.
  Assumes no prior familiarity with this specific project.
- **Deliverable:** A written survey at `notes/survey/memory-mechanisms-v1.md`
  (plus PDF), following the structure described in the last section of this
  brief.
- **Expected effort:** ~1–2 weeks, depending on depth. Aim for ~8–15 fully
  analyzed papers plus ~15–30 lighter one-liners.

> **If anything in this brief is ambiguous, stop and ask.** Misunderstandings
> early are cheap to fix. Misunderstandings after two weeks of searching are
> not.

---

## 1. Background — why this search exists

### 1.1 The research problem in one paragraph

We want AI agents that understand long videos (hour-scale content like a full
TV episode or a lecture) while only looking at a fraction of the frames —
reasoning as if they had seen every moment when they actually only looked at
a small subset. Rather than training a new model from scratch, we combine
existing models (VLMs, generative models, specialist vision models) through
an **agent** that orchestrates them. When the agent notices missing
information in what it has seen, it can call a specialist model to fill the
gap — for example, a depth estimator, a 3D reconstructor, or a generative
model that predicts what plausibly happened in unseen frames. Another
reasoning model then cleans up what came back. The supervisor of this
project (**Diwei Su**) calls this whole apparatus "memory" — the system
builds up an internal picture of the full video from sparse input, somewhat
like the human hippocampus lets people reason coherently from only brief
fixations.

### 1.2 The baseline we already picked

The project's baseline is **LongVideoAgent (ACL 2026)**. It's a published
long-video understanding agent that decides *which frames* of a video to
request next (smart temporal sampling trained with RL). Its repo ships
training and evaluation code plus data, so it's the anchor we'll modify
later. For this survey, you don't need to reproduce LongVideoAgent — just
know it exists and use its reference list + forward citations as seed
material.

### 1.3 Glossary — terms you'll see a lot

| Term | What it means in this project |
|---|---|
| **Agent / agentic** | A system that takes *multiple steps*, *calls tools*, or *iteratively queries* its inputs, rather than doing a single forward pass. Think "ReAct-style model that decides what to do next" or "multi-turn perception loop." |
| **Long video** | Video > 1 minute of duration. "Short" action-recognition benchmarks (UCF101, Kinetics, Something-Something) don't count. |
| **VLM / MLLM** | Vision-language model / multimodal LLM. Backbone models like LLaVA, Qwen-VL, InternVL, GPT-4V. These are what the agent usually calls. |
| **Memory** (Su's broad sense) | Any mechanism that lets the agent work with *less visual input than a naive approach* would need. Two flavors matter: (a) **retention** — storing what was already seen in a compact form; (b) **gap reconstruction** — filling in what was *not* seen via specialist models. Su uses the word "memory" for both flavors. Keep this broad definition in mind. |
| **LongVideoAgent** | Our baseline. ACL 2026. Smart frame-selection policy. See `literature/longvideoagent/`. |
| **VSI-Bench** | A spatial-reasoning benchmark. Relevant to the *project's origin*, not to this survey directly. You can ignore it unless a paper is explicitly evaluated on it. |
| **Ego4D / LVBench / VideoMME / EgoSchema / HourVideo** | Long-video benchmarks you're likely to see. Any of these counts as a "long-video evaluation." |

### 1.4 What we are NOT doing in this survey

- Not reproducing LongVideoAgent.
- Not building anything — this is pure literature survey.
- Not covering spatial reasoning (VSI-Bench, 3D scene understanding) as a
  standalone topic — the project has pivoted to long-video temporal reasoning.
- Not reviewing text-only LLM memory papers. MemGPT, Letta, generative
  agents — all *out* of scope unless they have a video evaluation.
- Not reviewing pre-2024 foundational work as primary material. Those get at
  most a short "background" mention.

---

## 2. What to look for — the tier definitions

The survey has two tiers. Do **all of Tier 1 first**, then Tier 2. Don't mix
them.

### 2.1 Tier 1 — memory mechanisms (primary)

A paper qualifies for Tier 1 if **all four** of the following hold:

1. **Timeframe:** published in **2024, 2025, or 2026**. If an arXiv paper
   straddles 2023/2024, take the first posting date.
2. **Domain:** the target is **video or vision**. If the paper is purely about
   text or tabular data, it's out, even if the memory mechanism is beautiful.
3. **Mechanism — at least one of these two flavors must be present:**
   - **(a) Retention memory** — there is an explicit module that stores
     what the system has already seen in a compact form. Examples of what
     "retention memory" looks like in practice:
     - An episodic buffer of past-frame features the agent can retrieve from
     - A "memory bank" of summaries, captions, or embeddings
     - A retrieval-augmented memory where frames are indexed and the agent
       fetches relevant ones
     - A memory-augmented transformer (external memory module attached to
       a transformer backbone)
     - A cognitive-map-style structure that tracks objects or spatial
       relations across time
   - **(b) Gap reconstruction** — the system *notices missing or insufficient
     input* and calls a specialist model to fill it in. Examples:
     - Sparse-frame video reasoning where the model predicts what missing
       frames would look like
     - A system that runs depth / 3D / segmentation estimators on demand to
       "see more" in an image
     - A generative model used as a prior to reconstruct information not
       directly observed
     - Any "tool-using" video agent that calls specialist models to fill
       perceptual gaps
4. **Agent-loop interface:** the mechanism must either
   - already operate agentically (tool-calling, multi-turn, iterative
     perception, ReAct-style), OR
   - be architecturally separable — i.e., the memory module has a clean
     interface that could plausibly plug into an agent built on top of a
     different backbone, without requiring us to retrain the whole thing.

PLUS: the paper evaluates on a **long-video task (>1 min)** OR the memory
mechanism is explicitly designed for **long-horizon** vision (e.g. hour-scale
egocentric video, multi-day robot episodes).

> **Rule of thumb:** if you can say, in one sentence, "this paper's memory
> trick could be grafted onto LongVideoAgent in a concrete way," it's Tier 1.

### 2.2 Tier 2 — long-video agentic, non-memory (secondary)

Start Tier 2 only after Tier 1 is done. A paper qualifies for Tier 2 if:

1. Published **2024–2026**.
2. Target task is **long-video understanding (>1 min)**.
3. Uses an **agentic design** — multi-step reasoning, tool-calling, iterative
   perception, or any non-monolithic-forward-pass architecture.
4. Does **not** already qualify for Tier 1 (no double-counting).

Tier 2 is lighter effort — short mini-cards, not full summaries.

### 2.3 Concrete examples (for calibration)

To help you calibrate what qualifies, here are invented-but-representative
examples of each bucket:

| Imagined paper | Bucket | Why |
|---|---|---|
| "EgoMemNet: retrieval-augmented memory bank for hour-long egocentric video QA," CVPR 2025 | **Tier 1 (a)** | Explicit retention memory module, long-video eval, clean module interface |
| "GapFiller: sparse-frame video agent that calls depth/generative priors to reconstruct unseen frames," NeurIPS 2025 | **Tier 1 (b)** | Gap-reconstruction flavor of memory, agentic design, long-video eval |
| "LongVideoReason: chain-of-thought reasoning for 30-min videos via tool-calling over a VLM," ICLR 2025 | **Tier 2** | Long-video, agentic, but has no memory module |
| "ActionClip-v3: contrastive pretraining for 8-second action recognition," CVPR 2024 | **Out** | Short video, no agent, no memory |
| "MemGPT: unbounded context for chat agents via paging," arXiv 2023 | **Out** | Text-only, no video eval, 2023 |
| "Neural Turing Machines revisited," NeurIPS 2024 | **Out** | Not video/vision |
| "MeMViT: memory-augmented multiscale ViT for long video," CVPR 2022 | **Out as primary (pre-2024); OK as background citation** | Qualifies conceptually but predates our window |

### 2.4 What "interesting" means — ranking signals

Within each tier, give more space to papers that have more of these
properties. These signals decide whether a paper gets a full card or a
one-liner.

1. **Code + data are publicly released.** Reproducibility is the path to
   actually using the idea later.
2. **Evaluated on hour-scale or near-hour-scale benchmarks** — Ego4D, LVBench,
   HourVideo, VideoMME-Long, MovieQA-long. Short clip benchmarks count less.
3. **Explicit comparison to prior memory mechanisms.** A paper that ablates
   multiple memory designs against each other is worth more than one that
   just adds "+memory" on top of a baseline.
4. **Citation adjacency to LongVideoAgent** — either cites it, is cited by it,
   or shares multiple references with it. These are the closest cousins to
   our chosen baseline.
5. **Clean module interface.** If the memory module is described as a
   standalone component with a defined input/output, it's more useful to us
   than one that's baked end-to-end into a specific backbone.

Papers that hit 3+ signals → full card. Papers that hit 0–2 signals → one-line
mini-card.

---

## 3. How to execute the search

### 3.1 Two-pass search plan

**Pass 1 — published venues (do this first):**

Search the proceedings of each of these venues for 2024, 2025, and 2026.

- **Machine learning / AI:** NeurIPS, ICML, ICLR
- **Computer vision:** CVPR, ICCV, ECCV, WACV
- **Natural language / multimodal:** ACL, EMNLP, NAACL (look for the
  multimodal and video tracks)
- **Journals:** TPAMI, IJCV

For each venue, scan paper titles and abstracts that match Tier 1 or Tier 2
keywords (next section). Don't try to read every paper — titles alone filter
out 90%.

**Pass 2 — arXiv sweep:**

After the venue pass, go to arXiv (categories `cs.CV`, `cs.AI`, `cs.LG`) and
do keyword searches over 2024–2026 preprints. This will catch work that's
recent enough to not be in a proceedings yet, or rejected-but-relevant work.

**Seeding (do this throughout):**

- Read the references section of LongVideoAgent. Follow anything that looks
  relevant.
- Read the references section of `notes/survey/agent-based-approaches-v2.md`.
  Follow anything relevant that isn't already in v2.
- For every Tier 1 paper you find, check its references and its forward
  citations (Google Scholar "cited by"). This is usually how you find the
  next 2–3 relevant papers.

### 3.2 Keyword lists

**Tier 1 keywords (try combinations of these):**

- "video agent" + "memory"
- "episodic memory" + "video"
- "long video" + "memory module"
- "memory-augmented" + "video"
- "cognitive map" + "video"
- "video question answering" + "memory"
- "video agent" + "reconstruction"
- "sparse-frame" + "video reasoning"
- "video gap filling"
- "specialist tools" + "video"
- "missing frame" + "video understanding"
- "hippocampus-inspired" + "video"
- "working memory" + "video agent"

**Tier 2 keywords:**

- "long video agent"
- "long video understanding" + "agent"
- "hour-long video" + "reasoning"
- "video tool use"
- "multi-step video reasoning"
- "iterative video perception"

### 3.3 Decision tree for each candidate paper

For every paper you encounter, walk through this in order:

```
1. Is it 2024–2026?                                 No → skip
2. Does it target video or vision?                  No → skip
3. Does it have either retention memory (a) or gap
   reconstruction (b), per §2.1.3?                  No → go to Tier 2 check
4. Does the mechanism plug into an agent loop?      No → skip
5. Is it evaluated on long-video (>1 min)?          No → skip
                                                    Yes → **Tier 1 candidate**

Tier 2 check:
1. Is it 2024–2026?                                 No → skip
2. Is the target task long-video understanding?     No → skip
3. Is it agentic (multi-step / tool-calling)?       No → skip
                                                    Yes → **Tier 2 candidate**
```

If you're unsure on any step, **write the paper down in a "pending" list
with a one-line note about why it's ambiguous**. Don't agonize. We'll review
the pending list together.

### 3.4 When to stop

Stop each tier when you're hitting diminishing returns:

- **Tier 1:** if you've queried 20 consecutive new sources (paper titles,
  arXiv listings, citation chains) and found fewer than 2 new qualifying
  papers, stop. Move to Tier 2.
- **Tier 2:** if you've queried 30 new sources and found fewer than 2 new
  qualifying papers, stop.

Both tiers combined should yield roughly 20–40 papers. If you're finding 100+,
your filter is too loose — come back and tighten.

---

## 4. How to record findings as you go

Keep a working file while you search — you don't need to wait until the end.
Suggested filename: `notes/survey/memory-mechanisms-v1-working.md` (or a
scratch subfolder).

### 4.1 Full-card template (Tier 1)

Use the project's standard summary card format, from
[`literature/SUMMARY_TEMPLATE.md`](../../literature/SUMMARY_TEMPLATE.md):

```markdown
# <Paper Title>

`<Venue Year>` · 🏛️ <Affiliation>

[👤 Authors](...) · [📄 Paper](...) · [💻 Code](...) · [📊 Dataset](...)

🏷️ **SUBJECT:** One-line category / domain framing.

❓ **PROBLEM:**
- 2–5 bullets: the gap this paper addresses

💡 **IDEA:** 1–2 sentences. **Bold** the key term.

🛠️ **SOLUTION:**
- **Component A:** one-line
- **Component B:** one-line
- **Component C:** one-line

🏆 **RESULTS:** 1–2 sentences with headline number + baseline.

💭 **THOUGHTS:** (optional)
- Relevance to Su's proposal — would this memory mechanism plug into
  LongVideoAgent? How?
- Any open questions or skepticism
```

### 4.2 Mini-card template (Tier 2)

```markdown
### <Paper Title> — `<Venue Year>`
[📄 Paper](...) · [💻 Code](...)

<2–3 sentence summary covering: what it does, how it's agentic, what
long-video benchmark it runs on.>
```

### 4.3 Per-paper metadata to capture regardless of tier

While reading, jot down for each paper:

- Title, authors, venue, year
- Paper link, code link, dataset link
- Benchmarks used (names matter — Ego4D vs EgoSchema vs LVBench)
- Average video length in the evaluation
- Does it call specialist models as tools? Which ones?
- Is there a memory module? What type (per §2.1.3)?
- Code release status (available / promised / not released)
- Whether it cites LongVideoAgent or vice versa

### 4.4 Pending / ambiguous list

Keep a list of papers you weren't sure how to classify, with one line each
explaining the ambiguity. Don't agonize over classification — surface the
decision instead.

---

## 5. Final deliverable structure

At the end, produce `notes/survey/memory-mechanisms-v1.md` with these
sections, in order:

1. **Header block**
   - Date, author, short abstract (~3 sentences describing the survey's
     scope and method — copy from §1.1 of this brief if you want)
   - Pointer back to this brief

2. **Tier 1 — Memory mechanisms in video agents**
   - One full card (§4.1) per Tier 1 paper, grouped by sub-flavor:
     - **1a. Retention memory** (memory banks, episodic buffers, retrieval,
       memory-augmented transformers, cognitive maps)
     - **1b. Gap reconstruction** (sparse-input reasoning, specialist-model
       orchestration, generative priors for unseen content)
   - Within each sub-flavor, order papers by how strongly they hit the
     ranking signals in §2.4

3. **Tier 2 — Long-video agentic (no memory)**
   - Mini-cards (§4.2), one per paper, in rough chronological order.

4. **Synthesis**
   - Taxonomy: what families of memory mechanism showed up? How do they
     relate to each other? A simple table or diagram is fine.
   - Which specific ideas from Tier 1 look like they could plug into
     LongVideoAgent's pipeline? Be concrete — name the paper, name the
     module, say how the interface would work.
   - Open gaps: what didn't anyone do? Where is the literature silent?

5. **Pending / unclassified**
   - The ambiguous-papers list from §4.4, if non-empty. Short explanation
     of why each is unclear.

6. **Background (pre-2024)**
   - A short subsection with 2–6 influential papers from before 2024 that
     the Tier 1 papers repeatedly cite (MeMViT, Long-term Feature Banks,
     etc.). One-line description each. This is context, not analysis.

7. **References**
   - Flat alphabetical list of everything cited.

Also produce a **PDF export** next to the Markdown file, matching the
convention of `notes/survey/agent-based-approaches-v2.pdf`.

---

## 6. Common ambiguities and how to handle them

- **"Is this paper really agentic, or just multi-step?"** If the paper has a
  loop where the system decides what to do next based on what it saw, that's
  agentic. A fixed multi-head forward pass is not. When in doubt — pending
  list.
- **"The paper has a memory module, but the backbone is huge and not
  swappable."** Still Tier 1 if the module itself has a clean interface that
  could in principle be lifted out. The point is whether the *idea* is
  portable.
- **"The paper uses 'memory' in the title but it's just a KV cache
  optimization."** Out. KV cache engineering is not a memory mechanism in
  our sense — there's no explicit representation of past or missing visual
  information.
- **"The paper reconstructs video for a different reason (e.g., video
  compression)."** Out, unless the reconstruction is part of an agent loop
  for understanding.
- **"The paper is on robotics / embodied AI, not video QA."** Tier 1 only if
  the memory mechanism is specifically about video input and transfers
  cleanly. Pure control/policy memory is out of scope.
- **"The paper was updated in 2024 but first posted in 2022."** Use the
  first-posting date. Out as primary, potentially OK as background.

When in genuine doubt, add to the pending list and keep moving. We'll resolve
ambiguities together in a review pass.

---

## 7. Checklist before handing the survey back

- [ ] ≥8 Tier 1 full cards
- [ ] All cards follow `SUMMARY_TEMPLATE.md` exactly (emojis, section order,
      length target)
- [ ] Synthesis section names specific papers + specific modules that could
      graft onto LongVideoAgent
- [ ] Pending list reviewed and either promoted to a tier or explicitly
      parked with a reason
- [ ] PDF exported alongside the Markdown
- [ ] References section is complete (no dangling citations)
- [ ] A skim of Tier 1 titles confirms they're all 2024–2026, video/vision,
      agentic, and have a memory component
