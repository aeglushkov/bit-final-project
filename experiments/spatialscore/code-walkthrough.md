# SpatialScore Code Walkthrough

Annotated notes from studying the SpatialScore codebase.

## Files to Study

1. [ ] `literature/spatialscore/code/test_qwen.py` — main evaluation script
2. [ ] `literature/spatialscore/code/utils/util.py` — answer extraction & metrics
3. [ ] `literature/spatialscore/code/SpatialAgent/agent.py` — agent loop
4. [ ] `literature/spatialscore/code/SpatialAgent/utils/prompt.py` — tool definitions + few-shot demos
5. [ ] `literature/spatialscore/code/SpatialAgent/utils/executor.py` — tool dispatch

---

## 1. test_qwen.py (308 lines)

Key functions:
- `load_model_and_components()` (line 20):
  - Q: what's the difference between tokenizer and processor?
- `process_model_input()` (line 29):
  - Describe the input and tell how the model should reply
- `generate_response()` (line 64):
  - takes assistant_prompt, img_paths, images and question and pass it to the model. 
  - Q: images variable is not used - what is it for?
- Evaluation logic (lines 166-218):
  - compares ground truth values and predictions

---

## 2. utils/util.py

*TODO: Add notes after studying*

---

## 3. SpatialAgent/agent.py (94 lines)

*TODO: Add notes after studying*

---

## 4. SpatialAgent/utils/prompt.py (~650 lines)

*TODO: Add notes after studying*

---

## 5. SpatialAgent/utils/executor.py

*TODO: Add notes after studying*
