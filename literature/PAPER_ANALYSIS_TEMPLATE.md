# Paper Analysis: [Title]

**Authors:** [Names] | **Venue:** [Conference/Journal, Year] | **Date read:** YYYY-MM-DD

**Links:** [Paper]() | [Code]() | [Project page]()

---

## 1. Core Contribution (2-3 sentences)

What does this paper do, and what is the single most important claim?

---

## 2. Relevance to Our Research

### 2.1 Relevance Rating: [High / Medium / Low]

Why is this paper relevant to building an agent layer that externalizes spatial reasoning from VLMs?

### 2.2 Key Takeaways for Our Direction

- What specific findings, methods, or insights from this paper can we use?
- Does it validate or challenge any assumptions in our approach?

### 2.3 Perception vs. Reasoning Evidence

Our core hypothesis: VLMs excel at perception but fail at spatial reasoning. Does this paper provide evidence for or against this?

| Aspect | Evidence from paper |
|--------|-------------------|
| Perception capability | |
| Reasoning capability | |
| Ego-allo transformation | |
| Error attribution | |

---

## 3. Method

### 3.1 Approach Summary

Brief description of the method/architecture/benchmark.

### 3.2 Architecture Diagram (if applicable)

Describe the pipeline or system. Note any modular components that separate perception from reasoning.

### 3.3 Spatial Representation

How does this paper represent spatial information? (e.g., cognitive maps, scene graphs, coordinate grids, natural language, embeddings, 3D point clouds)

| Representation type | Description | Strengths | Weaknesses |
|-------------------|-------------|-----------|------------|
| | | | |

---

## 4. Agent / Decomposition Aspects

*This section is specific to our research interest in agent-based approaches.*

### 4.1 Does the paper use any form of task decomposition?

- Single-pass vs. multi-step reasoning?
- Any chaining, tool use, or iterative querying?

### 4.2 Multi-turn VLM interaction

- Does the method query the VLM multiple times?
- Is there any feedback loop or self-correction?

### 4.3 External reasoning modules

- Are any reasoning steps handled outside the VLM? (code execution, geometric computation, explicit coordinate transforms, planning algorithms)
- If yes, what is delegated externally vs. kept inside the model?

### 4.4 Implications for our agent design

What specific design choices from this paper should we adopt, adapt, or avoid?

---

## 5. Evaluation

### 5.1 Benchmarks & Metrics

| Benchmark | Tasks | Metrics | Relevant to us? |
|-----------|-------|---------|-----------------|
| | | | |

### 5.2 Key Results

Highlight the most important numbers. Focus on results that reveal the perception-reasoning gap.

### 5.3 Baselines & Comparisons

What baselines matter? Are there agent-based baselines?

### 5.4 Failure Analysis

What types of errors dominate? Map to our error taxonomy:
- **Perception errors** — VLM fails to identify objects/regions
- **Egocentric-allocentric transformation errors** — VLM sees correctly but can't map to world frame
- **Relational reasoning errors** — VLM understands positions but fails at comparisons/distances
- **Integration errors** — information from multiple frames/views not combined correctly

---

## 6. Connections & Gaps

### 6.1 Related work this paper cites that we should read

| Paper | Why relevant |
|-------|-------------|
| | |

### 6.2 What this paper does NOT address (gaps we could fill)

- Limitations acknowledged by the authors
- Limitations we identify that relate to our agent approach

### 6.3 How this paper relates to others we've read

Connections, contradictions, or complementary findings with other papers in `literature/`.

---

## 7. Concrete Ideas Sparked

Actionable ideas for our agent architecture, experiments, or evaluation that this paper inspires. Be specific.

1. ...

---

## 8. Quick Reference

| | |
|---|---|
| **Problem type** | [benchmark / method / analysis / survey] |
| **Domain** | [indoor 3D / outdoor / general spatial / navigation / ...] |
| **Models involved** | |
| **Data** | |
| **Open-source code?** | [Yes/No — link] |
| **Spatial repr. used** | |
| **Agent/decomposition?** | [Yes/No — describe] |
| **Key number** | [The single most important result] |
