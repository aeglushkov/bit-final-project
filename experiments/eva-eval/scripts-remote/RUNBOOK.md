# Remote-server runbook

All long-running commands go through `launch.sh` so they survive SSH drops.
Each step's stdout+stderr land in `logs/<step>.log`.

## Initial setup (already done on `morgenshtern`)

Conda envs and weights are in place. Skip to "Run sequence" below.
If starting on a new box, follow `experiments/eva-eval/README.md` first.

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
| 3 | `03_start_servers` | Spin up Qwen on :18000 + InternVL2 on :18001 | 1-2 min |
| 4 | `05_smoke_phase5 [n=20]` | Run N questions through the full agent | 3-10 min |
| 5 | `06_full_phase2` | MASt3R on **all** 288 videos (resumable) | 6-10 hours |
| 6 | `07_full_phase3` | Build memory for **all** scenes (resumable) | 3-5 hours |
| 7 | `08_dev500` | 500-question stratified eval | 30-60 min |
| 8 | `09_full_eval [name]` | Full ~5000-question sweep | 10-15 hours |

Stop servers when done evaluating: `./experiments/eva-eval/scripts-remote/04_stop_servers.sh`

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
