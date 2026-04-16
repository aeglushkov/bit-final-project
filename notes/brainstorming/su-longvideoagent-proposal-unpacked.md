# Unpacking Su's 2026-04-16 proposal (LongVideoAgent + implicit frame compression)

- **Date:** 2026-04-16
- **Purpose:** Plain-language unpacking of each claim in Su's async reply, for someone
  not deeply familiar with video-processing conventions.
- **Source message:** [notes/meetings/2026-04-16.md](../meetings/2026-04-16.md)
- **Related:** [notes/survey/agent-based-approaches-v2.md](../survey/agent-based-approaches-v2.md),
  [notes/brainstorming/research-direction.md](research-direction.md)

---

## 1. "LongVideoAgent as baseline — baseline for what?"

Baseline for **long-video understanding** — specifically, answering questions about
long videos (hour-scale content, like full TV episodes). This is an important piece
of the pivot: Su is quietly switching the target task.

The original project was about *spatial reasoning* on VSI-Bench (short indoor
scenes, 3D geometry questions). LongVideoAgent is about *temporal reasoning over
long videos* (e.g. "In this TV episode, who did X before Y?"). Those are genuinely
different problems. Su doesn't flag the switch, but it's there.

He's picking LongVideoAgent as the anchor because it's the most *complete* package
available — the authors released the training code, the training data, and the
evaluation benchmarks. Starting from a working repo is cheaper than stitching one
together from multiple papers.

## 2. "Why is LongVideoAgent classified as 'when to look'?"

Think of long-video understanding as a two-part problem:

- **When to look** — a 1-hour video has maybe 100,000 frames. You can't feed them
  all to a VLM. So *which* frames do you pick? This is a temporal selection problem.
- **How to look** — once you've picked a frame (or a few), what do you do with it?
  Query it at low resolution? Zoom in? Run a depth estimator? Ask about geometry?
  This is a per-frame processing problem.

LongVideoAgent's main technical contribution is a **policy that decides which
frames/segments of the video to request next** — i.e., it's a smart frame-selector
trained with RL. The other papers in the survey mostly *assume the frames are
given* and focus on what to do with them (GCA does geometric reasoning, RieMind
does graph reasoning, VideoSeek zooms at different granularities, etc.).

So the split is basically: LongVideoAgent = smart sampler, everyone else = smart
per-frame processor.

## 3. "Humans need very limited information — why?"

This is not a "why" with a mathematical proof — it's an observation from cognitive
science that Su uses as **motivation**. A few facts in that neighborhood:

- Your eye doesn't sample continuously at a high rate. Effective temporal
  resolution is roughly 60 Hz in good conditions and much lower in practice
  (flicker fusion threshold).
- Humans use attention: you fixate on a few salient points per second and ignore
  everything else, yet you still build coherent understanding of a scene.
- Despite this sparse input, your spatial/temporal reasoning is stable — you don't
  need to see every moment of a movie to follow the plot.

Su's inference: *if humans can reason stably from sparse input, maybe models can
too, and we shouldn't assume "more frames = better."*

This is an **inspiration, not an argument**. It doesn't prove anything about what
VLMs need — it just motivates trying the direction.

## 4. "60-frame performance with only 30 frames — what does that even mean?"

Set the stage. When a VLM processes a video, it doesn't consume the raw video —
it samples a fixed number of frames (pictures) and feeds just those to the model.
Typical numbers:

- 8 frames, 16 frames, 32 frames — common for short-video VLMs
- 60, 128, 256 frames — common for long-video methods

More frames = more information available, but also more memory, more compute, and
slower inference. It's a direct cost.

So "get 60-frame performance with 30 frames" means:

> *Imagine the VLM currently achieves, say, 75% accuracy when you feed it 60
> frames. If you feed it only 30 frames, accuracy probably drops to, say, 68%.
> Su's question: can we design a system that takes only 30 frames as input but
> still scores 75%?*

**Why would you want this?** Compute. Halving the input roughly halves the memory
and compute cost of the vision encoder. At scale this matters — both for training
(you can train bigger models) and for deployment (cheaper inference).

The *trick* is: if you just drop frames, you lose information. So something has to
compensate. That something is point 5.

## 5. "Train a model that implicitly predicts the content of 60 frames from 30 — how?"

This is the actual technical proposal. Walking through it concretely:

**Standard training setup (no auxiliary loss):**
- Give the model 60 frames + a question
- Model outputs an answer
- Loss = how wrong the answer was
- Gradients update the model

**Su's proposed setup (with auxiliary loss):**
- Take 60 frames from the video
- Give the model only **30** of them (e.g., every other frame, or a random half)
- The model has **two heads**:
  1. **Answer head** — outputs the answer to the question (main task)
  2. **Prediction head** — outputs a guess of *what the missing 30 frames would
     have looked like*
- Loss = answer-wrongness + (how far off the predicted missing frames were from
  the real missing 30)
- Both losses flow gradients back into the shared encoder

**At inference time, the prediction head is discarded.** It's not needed. The
reason for training it was to *force the encoder to encode enough information in
its internal representation that it would have been able to reconstruct the
missing frames*.

Analogy: imagine a student studying for an exam. You can either (a) let them
memorize only the chapters that will be tested, or (b) make them also guess what's
in adjacent chapters. (b) forces deeper understanding of the material, even
though you never test the guessed chapters.

**How the "prediction" is implemented** depends on what you're predicting:
- Raw pixels? (like VideoMAE)
- Patch-level features in some embedding space? (like V-JEPA)
- Captions of the missing frames?
- Object trajectories?

Su's message is silent on this. **This is the single most important thing he
hasn't specified.** Each choice leads to a completely different implementation and
connects to different prior art.

Also: **this is a well-established idea.** It's called self-supervised video
pretraining with masked frame modeling. VideoMAE (2022) does almost exactly this:
mask 90% of video patches, train the model to reconstruct them. V-JEPA (Meta,
2024) does it in embedding space. Su's "internalization, not generation" phrasing
sounds novel, but mechanically it's the same recipe: add a reconstruction loss,
discard the decoder at inference.

## 6. "Modify LongVideoAgent in this direction"

Concretely, this means:
1. Take LongVideoAgent's existing architecture (vision encoder + policy + answer head).
2. Cut the input frame budget in half.
3. Add a prediction head + auxiliary loss for the missing frames.
4. Retrain on LongVideoAgent's existing training data.
5. Evaluate on LongVideoAgent's existing benchmarks.

The appeal is pragmatic: LongVideoAgent already ships data, training code, and
eval code. You don't start from scratch. You add one loss term, reduce input
size, retrain.

The catch: **LongVideoAgent is trained with RL (GRPO).** Adding a supervised
auxiliary loss to an RL-trained pipeline is not a clean plug-in — the two
objectives pull in different directions, and mixing them properly takes care.
Worth asking Su whether he has a concrete integration story or is hand-waving.

---

## Open questions worth raising with Su

1. **What exactly is being predicted?** Pixels, embeddings, captions, tracks?
   Each choice leads to very different prior art (VideoMAE vs V-JEPA vs captioning
   distillation vs world models).
2. **Target benchmark?** Is VSI-Bench / spatial reasoning paused, or does this
   still serve that thread?
3. **How is this different from masked video modeling (VideoMAE, V-JEPA)?** The
   "internalization vs generation" distinction describes what any encoder does
   under a reconstruction loss.
4. **Does the new mechanism address the original "pattern matching" critique?**
   A pixel-level reconstruction loss might even reinforce pixel shortcuts.
5. **Can LongVideoAgent's GRPO pipeline cleanly host a supervised auxiliary loss?**
   RL + reconstruction push in different directions.
6. **Evidence for the 30-frame claim?** Has he seen an ablation suggesting 2× is
   the right compression ratio, or is this just an intuition?
