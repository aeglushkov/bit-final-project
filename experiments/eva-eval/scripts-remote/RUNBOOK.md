# Remote-server runbook

All long-running commands go through `launch.sh` so they survive SSH drops.
Each step's stdout+stderr land in `logs/<step>.log`.

## Infrastructure (where the pipeline actually runs)

Two GPU servers. Access details (IP + login) live in `~/PycharmProjects/.morgen`
and `~/PycharmProjects/.neo` on the local workstation; both use `aleksandr` as
the SSH login.

| Server   | Hostname     | IP             | GPU      | VRAM   | What it hosts                          |
|----------|--------------|----------------|----------|--------|----------------------------------------|
| morgen   | morgenshtern | 185.71.171.85  | RTX 3090 | 24 GB  | full repo + all caches + all eval drivers |
| neo      | neo          | 185.71.171.86  | RTX 5090 | 32 GB  | bf16 InternVL2 server only (no full repo) |

- **morgen** is the primary box. The repo is at `~/github-projects/bit-final-project`,
  conda envs at `.conda/envs/{e-videoagent,mast3r,vllm,lmdeploy}`, HF caches at
  `.hf-cache/`, results at `results/`. Every eval driver and preprocessing step
  runs here. Start every workflow by SSH-ing to morgen.
- **neo** only runs the bf16 InternVL2 server when the split stack is active.
  Standalone miniconda at `~/miniconda3` with env `lmdeploy` (Python 3.10 + torch
  2.10 cu128 for Blackwell sm_120). Weights at `~/hf-cache/InternVL2-8B`. Server
  binds to `127.0.0.1:18001` only — never publicly exposed; morgen reaches it
  through an SSH tunnel opened by `_tunnel.sh`.
- **morgen → neo SSH** is passwordless (morgen's `~/.ssh/id_ed25519` is in neo's
  `authorized_keys`). Required by `03_start_servers_bf16.sh`, `04_stop_servers_bf16.sh`,
  and `_tunnel.sh`. If this breaks, regenerate by running `ssh-copy-id aleksandr@neo`
  from morgen.

### Three serving stacks (pick one — they share ports 18000/18001 on morgen)

1. **AWQ single-host** (default). `OpenGVLab/InternVL2-8B-AWQ` + `Qwen/Qwen2.5-7B-Instruct-AWQ`
   both on morgen's 3090 (~18 GB total). Start with `03_start_servers`, stop with
   `04_stop_servers`. The eval driver uses the default model entries in
   `config/models.yaml`; **no env vars needed**.

2. **bf16 split-host**. Full-precision Qwen on morgen + full-precision InternVL2
   on neo, joined by an SSH tunnel that maps morgen's `localhost:18001` to neo's
   loopback `:18001` so eval clients keep addressing `http://localhost:18001/v1`
   unchanged. Required because both bf16 LLMs (~40 GB combined) don't fit on
   either single card. Start with `03_start_servers_bf16`, stop with
   `04_stop_servers_bf16`. **The eval driver must set the bf16 model overrides**:

   ```bash
   export EVA_PLANNER=qwen2.5-7b-text-bf16
   export EVA_VLM=internvl2-8b-bf16
   ```

   Without these, the driver picks the default (AWQ) entries from `config/models.yaml`
   and sends requests to URLs that route to the wrong model names. The bf16 launchers
   already set these (`08_subset100_bf16.sh`).

3. **SenseNova-SI ("si") split-host** (Su's recommendation, 2026-05-14). Vanilla
   `OpenGVLab/InternVL3-8B` on morgen as the text planner + spatial-intelligence-
   enhanced `sensenova/SenseNova-SI-1.5-InternVL3-8B` on neo as the VLM, joined by
   the same SSH tunnel as the bf16 stack. The planner is multimodal-capable but
   used text-only (vLLM is launched with `--limit-mm-per-prompt image=0`). Start
   with `03_start_servers_si`, stop with `04_stop_servers_si`. Override the eval
   driver:

   ```bash
   export EVA_PLANNER=internvl3-8b-text-bf16
   export EVA_VLM=sensenova-si-1.5-internvl3-8b-bf16
   ```

   The SI launchers already set these (`08_subset100_si.sh`).

**Don't run more than one stack simultaneously** — they all collide on ports
18000/18001 and the split-host stacks' tunnel collides with AWQ lmdeploy on
neo's port. Always stop the current stack before starting another:

```bash
# bf16 -> AWQ
./experiments/eva-eval/scripts-remote/04_stop_servers_bf16.sh
./experiments/eva-eval/scripts-remote/_tunnel.sh down  # belt-and-suspenders
./experiments/eva-eval/scripts-remote/launch.sh 03_start_servers

# AWQ -> bf16
./experiments/eva-eval/scripts-remote/04_stop_servers.sh
./experiments/eva-eval/scripts-remote/launch.sh 03_start_servers_bf16

# bf16 -> SI (or AWQ -> SI: swap stop script)
./experiments/eva-eval/scripts-remote/04_stop_servers_bf16.sh
./experiments/eva-eval/scripts-remote/launch.sh 03_start_servers_si
```

### SI stack prerequisites

- **On morgen**: vanilla InternVL3-8B weights at `.hf-cache/InternVL3-8B`.
  `huggingface-cli download OpenGVLab/InternVL3-8B --local-dir .hf-cache/InternVL3-8B`.
- **On neo**: SenseNova-SI-1.5-InternVL3-8B weights at `~/hf-cache/SenseNova-SI-1.5-InternVL3-8B`.
  `huggingface-cli download sensenova/SenseNova-SI-1.5-InternVL3-8B --local-dir ~/hf-cache/SenseNova-SI-1.5-InternVL3-8B`.
- The neo lmdeploy env (`~/miniconda3/envs/lmdeploy`) is reused unchanged from the bf16 stack.
- Memory on morgen is tight: `--gpu-memory-utilization 0.85` puts InternVL3-8B at ~20 GB
  of the 3090's 24 GB. If vLLM OOMs at launch, drop to 0.80 and reduce `--max-model-len`
  from 16384.

### bf16 stack gotchas (learned the hard way; documented in commit `416f94b`)

- **`--backend pytorch`, not turbomind.** lmdeploy 0.13's turbomind backend crashes
  on InternVL2's nested `llm_config` dict before honoring `--trust-remote-code`.
- **`--trust-remote-code`** is required so transformers' `AutoConfig` loads InternVL2's
  custom config class.
- **Resize frames to 448×448 before sending.** Without that, lmdeploy expands each
  raw 512×384 frame into ~13 patches via `max_dynamic_patch=12`, producing ~26 K
  prompt tokens for 8 frames — beyond the effective KV-cache budget on the 5090
  (bf16 KV ≈ 0.5 MB/token; ~12 GB free for KV ≈ 24 K-token ceiling). Already done
  in `eva_eval/eval/baseline.py:_encode_image_b64`; the agent's frame tools should
  do the same if they start hitting `INPUT_LENGTH_ERROR`.
- **Pinned-IDs comparison.** The agent's stratified sampler filters by cached
  scenes *before* stratifying, so re-running with the same seed gives different IDs
  if more scenes have been cached since the prior run. To compare runs apples-to-apples,
  pass `--ids-file results/<prior-run>.ids.txt` to `scripts/03_run_vsibench.py`.
  The bf16 sample's IDs are pinned at `results/subset_bf16.ids.txt`.

## Initial setup (already done on `morgenshtern`)

Conda envs and weights are in place on both servers. Skip to "Run sequence" below.
If starting on a new box, follow `experiments/eva-eval/README.md` first. For a
new neo-equivalent (bf16 VLM host) the install steps are:

```bash
ssh aleksandr@<host>
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
~/miniconda3/bin/conda create -y -n lmdeploy python=3.10
~/miniconda3/envs/lmdeploy/bin/pip install --index-url https://download.pytorch.org/whl/cu128 "torch<=2.10" "torchvision<=0.25"
~/miniconda3/envs/lmdeploy/bin/pip install lmdeploy "transformers<5.0" timm einops "huggingface_hub[cli]" sentencepiece accelerate
~/miniconda3/envs/lmdeploy/bin/hf download OpenGVLab/InternVL2-8B --local-dir ~/hf-cache/InternVL2-8B
# Then from morgen: ssh-copy-id aleksandr@<host>
```

## Run sequence

```bash
ssh aleksandr@<remote>
cd ~/github-projects/bit-final-project
git pull --ff-only

# Local-only: tell git not to track downloaded data + caches
grep -q '^data/' .git/info/exclude || echo 'data/' >> .git/info/exclude
grep -q '^results/' .git/info/exclude || echo 'results/' >> .git/info/exclude
grep -q '^logs/' .git/info/exclude || echo 'logs/' >> .git/info/exclude
grep -q '^cache/' .git/info/exclude || echo 'cache/' >> .git/info/exclude
```

Steps run in order. Each ends with `DONE_<NAME>` in its log when finished.

| # | Script | What it does | Expected runtime |
|---|---|---|---|
| 0 | `00_video_download` | Pull 3 zips from HF + extract → `data/vsibench-videos/` (~50 GB) | 30-60 min |
| 1 | `01_smoke_phase2 [video.mp4]` | MASt3R-SfM on 1 video → `cache/vsibench/<id>/` | 5-10 min |
| 2 | `02_smoke_phase3 [scene_id]` | Build ObjectMemory for 1 scene | 1-2 min |
| 3 | `03_start_servers` | Spin up AWQ Qwen on :18000 + AWQ InternVL2 on :18001 (single-host morgen) | 1-2 min |
| 3b | `03_start_servers_bf16` | bf16 Qwen on morgen + bf16 InternVL2 on neo, joined by SSH tunnel | 3-5 min |
| 3c | `03_start_servers_si` | bf16 InternVL3-8B planner on morgen + SenseNova-SI-1.5 VLM on neo, joined by SSH tunnel | 3-5 min |
| 4 | `05_smoke_phase5 [n=20]` | Run N questions through the full agent | 3-10 min |
| 5 | `06_full_phase2` | MASt3R on **all** 288 videos (resumable) | 6-10 hours |
| 6 | `07_full_phase3` | Build memory for **all** scenes (resumable) | 3-5 hours |
| 7 | `08_dev500` | 500-question stratified eval | 30-60 min |
| 8 | `09_full_eval [name]` | Full ~5000-question sweep | 10-15 hours |
| 9 | `10_baseline [n] [vlm] [nfr]` | Raw-VLM baseline on lmdeploy (AWQ INT4) | 30 min |
| 10 | `10b_authors_baseline [n]` | Raw-VLM baseline on authors' unmodified lmms-eval (full-precision bf16, native frame template); needs `.conda/envs/vsibench` from `literature/thinking-in-space/CLAUDE.md` install steps | 30-60 min |

Stop servers when done evaluating:
- AWQ stack: `./experiments/eva-eval/scripts-remote/04_stop_servers.sh`
- bf16 split stack: `./experiments/eva-eval/scripts-remote/04_stop_servers_bf16.sh`
- SI split stack: `./experiments/eva-eval/scripts-remote/04_stop_servers_si.sh`

To run evaluations against the bf16 split stack, set the model overrides on
the eval driver before invoking 05/08/09:
```bash
export EVA_PLANNER=qwen2.5-7b-text-bf16
export EVA_VLM=internvl2-8b-bf16
```

For the SenseNova-SI stack, use:
```bash
export EVA_PLANNER=internvl3-8b-text-bf16
export EVA_VLM=sensenova-si-1.5-internvl3-8b-bf16
```

## Launching anything

```bash
./experiments/eva-eval/scripts-remote/launch.sh 00_video_download
./experiments/eva-eval/scripts-remote/launch.sh 01_smoke_phase2
./experiments/eva-eval/scripts-remote/launch.sh 02_smoke_phase3
./experiments/eva-eval/scripts-remote/launch.sh 03_start_servers
./experiments/eva-eval/scripts-remote/launch.sh 05_smoke_phase5 20
./experiments/eva-eval/scripts-remote/launch.sh 06_full_phase2
./experiments/eva-eval/scripts-remote/launch.sh 07_full_phase3
./experiments/eva-eval/scripts-remote/launch.sh 08_dev500
./experiments/eva-eval/scripts-remote/launch.sh 09_full_eval qwen_internvl2
```

`launch.sh` prints the PID and a `tail -f` command for the log. Then disconnect freely.

## Watching

```bash
tail -f logs/<step>.log                       # live
grep -E "DONE_|ERROR|Traceback" logs/*.log     # quick scan
pgrep -af <step>                               # still running?
```

## Stopping

```bash
pkill -SIGINT -f <step>     # gentle
pkill -9 -f <step>          # force
./experiments/eva-eval/scripts-remote/04_stop_servers.sh    # vllm + lmdeploy
```

## Recovery patterns

- **Phase 2/3 video failed** — script exits non-zero with details in
  `cache/vsibench/preprocess_failures.jsonl` or `memory_failures.jsonl`.
  Re-running skips already-done items.
- **Eval crashed mid-run** — `results/<name>.jsonl` is partial. Re-launch
  with a different `--output` name and merge later, OR fix the cause and
  rerun from scratch (the runner overwrites by default).
- **Server OOM** — `04_stop_servers` then `03_start_servers`. Lower
  `--gpu-memory-utilization` or `--cache-max-entry-count` if it keeps OOMing.
- **Disk filling up** — videos are in `data/`, model caches in `.hf-cache/`.
  Both are gitignored. `du -sh data/ .hf-cache/ cache/` to see hotspots.

## Clean uninstall

```bash
cd ~/github-projects && rm -rf bit-final-project/
```

Removes everything: code, envs, caches, downloads, results.
