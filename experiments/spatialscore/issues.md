# Known Issues — SpatialAgent on SpatialScore

## Issue #2 (SERIOUS): `extract_option()` misparses `<answer>` tags

The CoTAPrompt instructions (prompt.py:327) tell the model to wrap answers in `<answer>B</answer>`. But `extract_option()` in `utils/util.py:225-278` runs pattern matching on the original text (with tags) instead of the cleaned text. The fallback regex `re.search(r'[A-Fa-f]', text)` finds the `a` in `<answer>` and returns `'A'` instead of the correct letter.

**Impact:** Every multi-choice answer using `<answer>` tags gets corrupted. 57/110 samples are multi-choice.

**Fix:** After stripping tags on line 238 (`clean_text`), run all subsequent pattern matching on `clean_text` instead of `text`. Or strip `<answer>`/`</answer>` from `final_answer` in `run_agent.py` before passing to `evaluate_answer()`.

---

## Issue #3 (MODERATE): Agent exits without answer when max_steps hit

When the 3B model can't follow the complex JSON format and never calls Terminate, `final_answer` stays `None`. `run_agent.py:337` defaults to `pred_answer = ""` which is guaranteed wrong.

**Impact:** Unknown without logs. Likely frequent with 3B model given the massive system prompt (9 tools, 7 few-shot examples, strict JSON).

**Fix:** When `final_answer` is None, extract an answer from the last assistant message in the conversation history as a fallback.

---

## Issue #4 (MINOR): Initial message doesn't declare available images

`get_prompt_for_curr_query()` (prompt.py:346-348) doesn't tell the model which images are available. The few-shot examples reference `image-0`, `image-1` but many real questions just say "the image". The model can't know to use `image-0` in tool call arguments.

**Fix:** Prepend to the question: `"The following input images are provided: image-0, image-1, ..."`
