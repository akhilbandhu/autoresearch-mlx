# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled entirely through `program.md`. This fork preserves every design rule — 5-minute wall-clock budget, single mutable `train.py`, one metric (`val_bpb`), keep/revert via git — and runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch or CUDA required.
Huge shout out as well to [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) for actually doing the MLX implementation work that helped prove this direction out.

## Quick start

Requirements: Apple Silicon Mac (M1/M2/M3/M4), Python 3.10+, uv.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (one-time)
uv run prepare.py

# Run a single training experiment
uv run train.py

# Start autonomous research
# Point Claude Code (or any agent) at program.md and let it go
```

The checked-in defaults aim to fit 16 GB Apple Silicon Macs by keeping `FINAL_EVAL_BATCH_SIZE` conservative. If you have more unified memory and want faster evaluation, raise that value in `train.py`.

## Multi-agent tmux launcher

If you want to run several autoresearch agents in parallel and watch them live, use `scripts/launch_tmux_agents.py`. It creates one git worktree per agent under a sibling directory, initializes each worktree with your local baseline in `results.tsv`, and launches one tmux window per agent with a selectable harness.

```bash
# 1. Review and edit the example profile
$EDITOR scripts/agents.example.json

# 2. Preview the worktrees/session that would be created
uv run python scripts/launch_tmux_agents.py --config scripts/agents.example.json --dry-run

# 3. Launch the swarm
uv run python scripts/launch_tmux_agents.py --config scripts/agents.example.json

# 4. Attach to the tmux session
tmux attach -t autoresearch-mar8
```

Notes:

- Each agent gets its own branch/worktree, so commits and resets do not interfere.
- The example profile shows mixed harnesses: Copilot CLI plus Codex CLI.
- For Copilot agents you can pick different underlying models (`claude-sonnet-4.5`, `gpt-5.4`, `gemini-3-pro-preview`, etc.).
- For Codex agents you can either rely on your local default model or set `"model"` explicitly if your Codex setup supports it.
- Parallel agents consume premium requests quickly, so start with 2-3 agents and scale up once the workflow looks healthy.
- If the base checkout is dirty, the launcher warns you and still branches from committed `HEAD`.
- Built-in harnesses are currently `copilot` and `codex`. If you want Claude CLI, Pi, or something else, use the `custom` harness with a command template.

### How the multi-agent launcher works

1. You define a run profile in `scripts/agents.example.json` with a `run_tag`, a shared local baseline, and a list of agents.
2. For each agent, the launcher creates a separate git worktree and branch under a sibling directory such as `../autoresearch-mlx-worktrees/mar8/sonnet45/`.
3. Each worktree gets your local baseline appended to its own `results.tsv`, so every agent compares against the same known-good run on your Mac.
4. The launcher builds a prompt from `program.md`, the local baseline numbers, and each agent's `extra_prompt` so the agents can pursue slightly different research strategies.
5. It starts a tmux session with an `overview` window plus one window per agent, making it easy to watch progress live.
6. Inside each agent window, it runs the selected harness: for example `copilot --experimental --autopilot --model gpt-5.4 -p "<prompt>"` or `codex exec --full-auto "<prompt>"`.
7. If you set `"harness": "custom"`, the launcher will run your own command template, so you can plug in other local CLIs such as Claude Code or Pi as long as they accept a prompt non-interactively.
8. Each window also writes a persistent log file under `../autoresearch-mlx-worktrees/logs/<run_tag>/`, so you can inspect output without staying attached to tmux.
9. Because each agent has its own branch and worktree, it can commit, amend, and reset independently without clobbering the others.

### Harness configuration

Each agent in `scripts/agents.example.json` can choose its runner independently:

- `"harness": "copilot"` uses GitHub Copilot CLI prompt mode.
- `"harness": "codex"` uses `codex exec`.
- `"harness": "custom"` runs your own command template.

For custom harnesses, provide a `"command"` array and use placeholders such as `{prompt}`, `{model}`, `{worktree}`, `{log_path}`, and `{name}`. Example:

```json
{
  "name": "wrapped-agent",
  "harness": "custom",
  "model": "sonnet",
  "command": ["python", "scripts/my-harness-wrapper.py", "{prompt}"],
  "extra_prompt": "Focus on simpler optimizer changes first."
}
```

That wrapper can invoke Claude, Pi, or any other local agent CLI with whatever flags it expects.

## How it works

Same as the original. Three files that matter:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation. Not modified.
- **`train.py`** — model, optimizer, training loop. The agent edits this.
- **`program.md`** — agent instructions. Point your agent here.

The agent reads `program.md`, modifies `train.py`, runs a 5-minute experiment, checks `val_bpb`, and commits or reverts. Repeat overnight. Wake up to results.

## Results on M1 Mac Studio (48GB)

Starting from the upstream default configuration and running the autoresearch loop:

| Experiment | Change | val_bpb | Action |
|---|---|---|---|
| baseline | default config | 2.667 | keep |
| 1 | halve batch size | 2.589 | keep |
| 2 | 10x matrix LR | 2.534 | keep |
| 3 | depth 8 → 4 | 1.808 | keep |

Key finding: Apple Silicon throughput in a 5-minute window favors smaller, faster-training models. The autoresearch loop discovered this automatically — more optimizer steps beat more parameters when compute time is fixed.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon, unified memory.
- **AdamW only.** Muon optimizer port is future work.
- **Smaller eval token budget.** Reduced for faster iteration, while still using the same `evaluate_bpb` function from `prepare.py`.
- **16 GB-safe default eval batch.** The checked-in `FINAL_EVAL_BATCH_SIZE` is conservative enough for lower-memory Apple Silicon Macs; larger-memory machines can increase it for faster evaluation.
- **MFU reporting is placeholder.** No Apple Silicon FLOPs benchmark exists equivalent to H100_BF16_PEAK_FLOPS. `peak_vram_mb` reports MLX unified memory via `mx.metal.get_peak_memory()`.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — early MLX autoresearch implementation
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. Original copyright preserved. See [LICENSE](LICENSE).
