# SpatialScore Experiment Findings

## Overview

Comparison of Qwen2.5-VL-3B evaluated in two modes:
- **Baseline (MLLM only):** Direct VLM inference, no tool use
- **SpatialAgent:** Agentic loop with tool-augmented spatial reasoning

## Overall Results

| Mode | Accuracy | Correct | Total |
|---|---|---|---|
| Baseline | 51.82% | 57 | 110 |
| SpatialAgent | 10.91% | 12 | 110 |
| **Delta** | **-40.91%** | | |

## By Source

| Source | Baseline Acc | Agent Acc | Delta | Baseline N | Agent N |
|---|---|---|---|---|---|
| 3DSRBench | 40.0% | 80.0% | +40.0% | 10 | 10 |
| BLINK | 50.0% | 10.0% | -40.0% | 10 | 10 |
| MMVP | 50.0% | 0.0% | -50.0% | 10 | 10 |
| QSpatialBench-Plus | 40.0% | 0.0% | -40.0% | 10 | 10 |
| QSpatialBench-ScanNet | 60.0% | 10.0% | -50.0% | 10 | 10 |
| RealWorldQA | 50.0% | 20.0% | -30.0% | 10 | 10 |
| SpatialSense | 60.0% | 0.0% | -60.0% | 10 | 10 |
| VGBench | 20.0% | 0.0% | -20.0% | 10 | 10 |
| VSR-ZeroShot | 70.0% | 0.0% | -70.0% | 10 | 10 |
| cvbench | 50.0% | 0.0% | -50.0% | 10 | 10 |
| spatialbench | 80.0% | 0.0% | -80.0% | 10 | 10 |

## By Category

| Category | Baseline Acc | Agent Acc | Delta | Baseline N | Agent N |
|---|---|---|---|---|---|
| 3D Positional Relation | 65.0% | 0.0% | -65.0% | 20 | 20 |
| Counting | 60.0% | 3.3% | -56.7% | 30 | 30 |
| Depth and Distance | 40.0% | 0.0% | -40.0% | 10 | 10 |
| Object Localization | 20.0% | 0.0% | -20.0% | 10 | 10 |
| Object Properties | 50.0% | 45.0% | -5.0% | 20 | 20 |
| Others | 50.0% | 10.0% | -40.0% | 20 | 20 |

## By Question Type

| Question Type | Baseline Acc | Agent Acc | Delta | Baseline N | Agent N |
|---|---|---|---|---|---|
| judgment | 61.9% | 0.0% | -61.9% | 21 | 21 |
| multi-choice | 43.9% | 15.8% | -28.1% | 57 | 57 |
| open-ended | 59.4% | 9.4% | -50.0% | 32 | 32 |

## Per-Sample Analysis

Out of 1 samples compared:
- **Both correct:** 0
- **Both wrong:** 1
- **Agent improved (baseline wrong → agent correct):** 0 samples
- **Agent regressed (baseline correct → agent wrong):** 0 samples

### Improved Sample IDs
None

### Regressed Sample IDs
None

## Key Takeaways

1. **Overall delta:** SpatialAgent does not improve over the baseline by 40.9 percentage points.
2. **Categories with largest improvement:** (see table above)
3. **Categories where agent hurts:** (see table above)
4. The SpatialAgent paper reports +6-8% improvement with Qwen3-VL-4B/8B models. Our results with Qwen2.5-VL-3B differ from this trend.

## Implications for Our Research

- Categories where the agent improves most suggest where **externalized spatial reasoning** adds the most value.
- Categories where it regresses suggest the agent's tool calls may introduce noise or the 3B model struggles to follow the agentic format.
- Object Localization and Depth/Distance tasks are prime candidates for tool-augmented approaches.
