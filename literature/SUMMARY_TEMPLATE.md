# Paper `summary.md` Template

Every `literature/<paper>/summary.md` must follow the structure below. The goal
is a single scannable card for fast paper triage — deeper method/dataset/
training detail belongs in `analysis.md`, not here.

## Template

```markdown
# <Paper Title>

`<Venue Year>` · 🏛️ <Affiliation 1> · 🏛️ <Affiliation 2>

[👤 Authors](<author-or-first-author-link>) · [📄 Paper](<arxiv-or-pdf>) · [💻 Code](<repo>) · [📊 Dataset](<dataset-link>) · [🚀 Demo](<project-page>)

🏷️ **SUBJECT:** One-line category / domain framing.

❓ **PROBLEM:**
- Bullet 1 — concise gap or failure mode
- Bullet 2
- Bullet 3

💡 **IDEA:** One-to-two-sentence core insight. **Bold** the key novel term.

🛠️ **SOLUTION:**
- **Component A:** one-line description
- **Component B:** one-line description
- **Component C:** one-line description

🏆 **RESULTS:** One-to-two sentences with the headline number(s) and the comparison baseline.

<!-- Optional, only when there are genuine open questions -->
💭 **THOUGHTS:**
- **Open question 1:** …
- **Open question 2:** …
```

## Rules

- **Venue tag** uses backticks for a chip-like look: `` `Arxiv 2026` ``,
  `` `ICLR 2026` ``, `` `CVPR 2025` ``. Use `Arxiv <year>` for unpublished work.
- **Affiliations** are prefixed with 🏛️ and separated by `·`. Omit the row if
  unknown.
- **Links row** shows only the links that actually exist — do not leave broken
  placeholders. Common link types: 👤 Authors, 📄 Paper, 💻 Code, 📊 Dataset,
  🚀 Demo / Project.
- **Section emojis are fixed:** 🏷️ ❓ 💡 🛠️ 🏆 💭. Keep the ALL-CAPS bold
  labels — they are the visual anchors for scanning.
- **Section formatting:**
  - `SUBJECT`: one line, prose.
  - `PROBLEM`: 2–5 bullets.
  - `IDEA`: 1–2 sentences of prose, with the key novel term bolded.
  - `SOLUTION`: 3–6 bullets, each a `**Component:**` line.
  - `RESULTS`: 1–2 sentences of prose with the headline number(s).
  - `THOUGHTS` (optional): include only when there are genuine open questions
    or research reactions; omit the header entirely when empty.
- **Length target:** ~20–45 lines. If you need more space, it belongs in
  `analysis.md`.
- **Nothing else.** No trailing citation blocks, no training hyperparameters,
  no full author lists, no ablation tables — those live in `analysis.md`.
