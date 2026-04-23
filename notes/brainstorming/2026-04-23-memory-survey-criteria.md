# Criteria for the agentic long-video memory survey

- **Date:** 2026-04-23
- **Status:** Criteria locked. Deep research to follow; output will go to `notes/survey/`.
- **Related:**
  [notes/meetings/2026-04-16.md](../meetings/2026-04-16.md),
  [notes/meetings/2026-04-17.md](../meetings/2026-04-17.md),
  [notes/brainstorming/su-longvideoagent-proposal-unpacked.md](su-longvideoagent-proposal-unpacked.md),
  [notes/survey/agent-based-approaches-v2.md](../survey/agent-based-approaches-v2.md)

## What we're researching (plain words)

We want AI agents that understand long videos while only looking at a fraction
of the frames — reasoning as if they saw 60 when they only got 30. Instead of
training a new model from scratch, we combine existing models through an
agent: when the agent notices missing information, it calls a specialist model
to fill in the gaps (2D, 3D, depth, segmentation), then calls a reasoning
model to clean up the result. Su calls this "memory" — the agent ends up
internalizing the full picture from sparse input, similar to how the human
brain reasons stably without seeing every moment. LongVideoAgent is our
baseline.

## Tier 1 — primary (memory mechanisms)

A paper qualifies if ALL of the following hold:

- Published **2024–2026**
- Target domain is **video or vision** (not text-only, not tabular)
- Contains a mechanism where the agent works with **less visual input than a
  naive approach would need**, via one or both of:
  - **(a) Retention memory** — episodic buffer, memory bank,
    retrieval-augmented memory, memory-augmented transformer,
    cognitive-map-style state
  - **(b) Gap reconstruction** — detecting missing/sparse input and filling
    it via specialist models (depth, 3D, segmentation, generative priors),
    reconstructing what wasn't observed
- The mechanism has a **clean interface into an agent loop** — either already
  agentic (tool-calling, multi-turn, iterative perception) or architecturally
  separable from the backbone
- Evaluates on **long-video (>1 min)** OR the mechanism is explicitly designed
  for long-horizon vision

## Tier 2 — secondary (long-video agentic, non-memory)

Run only after Tier 1 is exhausted. A paper qualifies if:

- Published 2024–2026
- **Long-video understanding (>1 min)** is the target task
- Uses an **agentic / multi-step / tool-calling / iterative-perception** design
- Does NOT already qualify for Tier 1 (no double-counting)

## Exclusions (explicit)

- Short-video-only methods (<1 min targets, e.g. action recognition)
- Pure text-memory LLM-agent papers (MemGPT, Letta, Voyager) unless evaluated
  on video
- Pre-2024 work — acknowledged as background citations only, no cards
- VLM pretraining / scaling papers with no agent or memory story
- Neuroscience theory papers with no computational instantiation

## Ranking signals (full card vs. one-liner)

Within each tier, prioritize papers that have more of these properties:

1. **Code + data released** — reproducibility → viable baseline transfer
2. Evaluated on **hour-scale / Ego4D / LVBench-scale** benchmarks
3. Explicit comparison to prior memory mechanisms (not just a "+memory"
   ablation)
4. Citation adjacency to **LongVideoAgent** (cites it, or cited by it)

## Search strategy

**Pass 1 — well-known venues first (2024–2026):**
- ML / AI: NeurIPS, ICML, ICLR
- CV: CVPR, ICCV, ECCV
- NLP: ACL, EMNLP, NAACL (for multimodal-agent crossovers)
- Journals: TPAMI, IJCV

**Pass 2 — arXiv sweep:**
- Categories: cs.CV, cs.AI, cs.LG
- Preprints 2024–2026 matching Tier 1/Tier 2 keywords

**Seeding:**
- References and forward-citations of LongVideoAgent
- References in `notes/survey/agent-based-approaches-v2.md`

**Tier 1 keywords:**
"video agent memory", "episodic memory video", "long video memory module",
"memory-augmented video agent", "video question answering memory",
"cognitive map video", "video agent reconstruction", "sparse-frame video
reasoning", "video gap filling agent", "specialist vision tools video",
"missing frame video understanding"

**Tier 2 keywords:**
"long video agent", "long video understanding agent", "hour-long video
reasoning", "video tool use"

**Stop condition:**
- Tier 1: fewer than 2 new qualifying papers per 20 queried
- Tier 2: fewer than 2 new qualifying papers per 30 queried

## Deliverable (downstream)

Target file: `notes/survey/memory-mechanisms-v1.md` (plus PDF export, matching
the `agent-based-approaches-v2` convention).

Structure:
- **Tier 1 section** — one card per paper following the
  [`literature/SUMMARY_TEMPLATE.md`](../../literature/SUMMARY_TEMPLATE.md)
  5-section emoji format (🏷️ SUBJECT / ❓ PROBLEM / 💡 IDEA / 🛠️ SOLUTION /
  🏆 RESULTS, optional 💭 THOUGHTS)
- **Tier 2 section** — 2–3 sentence mini-cards
- **Synthesis section** — taxonomy of memory mechanisms encountered, which
  ideas plausibly transfer to LongVideoAgent's pipeline, open gaps in the
  literature

Survey-level exit criteria (to be checked when the survey runs, not now):
- ≥8 Tier 1 cards
- Synthesis explicitly positions each Tier 1 paper relative to Su's proposal
- Explicit shortlist of memory-mechanism designs that could plug into
  LongVideoAgent's pipeline
