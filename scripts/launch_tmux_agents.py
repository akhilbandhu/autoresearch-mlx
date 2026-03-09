#!/usr/bin/env python3
"""
Launch parallel autoresearch agents in tmux, one worktree per agent.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path


COPILOT_TRAILER = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

DEFAULT_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    Read @program.md, @README.md, @prepare.py, @train.py, @results.tsv, and @scripts/run_train_with_lock.py.
    You are inside a dedicated git worktree and branch for this agent only.
    Follow @program.md closely and continue the autonomous experiment loop from here.

    Local baseline for this Mac:
    - val_bpb: {baseline_val_bpb:.6f}
    - memory_gb: {baseline_memory_gb:.1f}

    Hard rules:
    - modify only train.py
    - commit, amend, and reset only inside this worktree/branch
    - do not touch sibling worktrees or the parent checkout
    - keep logging results to results.tsv
    - this machine is shared by multiple agent worktrees, so never run `uv run train.py` directly
    - for every experiment run, always use `uv run python scripts/run_train_with_lock.py > run.log 2>&1`
    - continue experimenting until interrupted; do not stop to ask the human

    Guidance:
    - start by reading the files above and checking git status/results.tsv
    - compare only against this machine's baseline and any later kept results in this branch
    - prefer simple MLX-native ideas before adding complexity
    - if an experiment crashes, debug briefly, then move on if the idea is bad
    - reading, editing, and reasoning can happen in parallel across agents, but training must stay serialized

    Agent-specific focus:
    {extra_prompt}
    """
)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
    )


def git(repo_root: Path, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=cwd or repo_root, check=check)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def shell_join(parts: list[str]) -> str:
    return " ".join(shell_quote(part) for part in parts)


def validate_string_list(field_name: str, value: object, *, required: bool = False) -> list[str]:
    if value is None:
        if required:
            raise SystemExit(f"Missing required list field: {field_name}")
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"Field {field_name!r} must be a list of strings.")
    return value


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def resolve_worktree_root(repo_root: Path, config: dict) -> Path:
    root = Path(config.get("worktree_root", "../autoresearch-mlx-worktrees")).expanduser()
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    return root


def tmux_session_exists(session_name: str) -> bool:
    proc = run(["tmux", "has-session", "-t", session_name], check=False)
    return proc.returncode == 0


def ensure_worktree(repo_root: Path, base_ref: str, branch_name: str, worktree_path: Path) -> tuple[bool, str]:
    branch_exists = git(repo_root, "show-ref", "--verify", f"refs/heads/{branch_name}", check=False).returncode == 0
    if worktree_path.exists():
        proc = git(repo_root, "branch", "--show-current", cwd=worktree_path)
        current_branch = proc.stdout.strip()
        if current_branch != branch_name:
            raise SystemExit(
                f"Existing worktree at {worktree_path} is on branch {current_branch!r}, expected {branch_name!r}."
            )
        return False, current_branch

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if branch_exists:
        git(repo_root, "worktree", "add", str(worktree_path), branch_name)
    else:
        git(repo_root, "worktree", "add", "-b", branch_name, str(worktree_path), base_ref)
    return True, branch_name


def ensure_baseline_commit(
    repo_root: Path,
    worktree_path: Path,
    base_commit: str,
    baseline: dict,
    agent_name: str,
) -> bool:
    results_path = worktree_path / "results.tsv"
    if not results_path.exists():
        raise SystemExit(f"Missing results.tsv in {worktree_path}")

    row = (
        f"{base_commit}\t{baseline['val_bpb']:.6f}\t{baseline['memory_gb']:.1f}\tkeep\t"
        f"{baseline['description']}"
    )
    current = results_path.read_text()
    if row in current:
        return False

    if current and not current.endswith("\n"):
        current += "\n"
    current += row + "\n"
    results_path.write_text(current)
    git(repo_root, "add", "results.tsv", cwd=worktree_path)
    git(
        repo_root,
        "commit",
        "-m",
        f"baseline: record {agent_name}",
        "-m",
        COPILOT_TRAILER,
        cwd=worktree_path,
    )
    return True


def build_prompt(agent: dict, baseline: dict) -> str:
    extra_prompt = agent.get("extra_prompt", "").strip() or "Use your own judgment and keep exploring."
    return DEFAULT_PROMPT_TEMPLATE.format(
        baseline_val_bpb=baseline["val_bpb"],
        baseline_memory_gb=baseline["memory_gb"],
        extra_prompt=extra_prompt,
    )


def get_agent_harness(agent: dict) -> str:
    return agent.get("harness", "copilot")


def get_agent_model_display(agent: dict) -> str:
    return agent.get("model", "default")


def format_parts(parts: list[str], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in parts]


def build_copilot_command(agent: dict, prompt: str, worktree_path: Path, log_path: Path) -> list[str]:
    model = agent.get("model")
    max_continues = str(agent.get("max_autopilot_continues", 200))
    command = [
        "copilot",
        "--experimental",
        "--autopilot",
        "--allow-all",
        "--no-ask-user",
        "--max-autopilot-continues",
        max_continues,
    ]
    if model:
        command.extend(["--model", model])
    values = {
        "prompt": prompt,
        "worktree": str(worktree_path),
        "log_path": str(log_path),
        "name": agent["name"],
        "model": agent.get("model", ""),
    }
    command.extend(format_parts(validate_string_list("harness_args", agent.get("harness_args")), values))
    command.extend(["-p", prompt])
    return command


def build_codex_command(agent: dict, prompt: str, worktree_path: Path, log_path: Path) -> list[str]:
    model = agent.get("model")
    command = [
        "codex",
        "exec",
        "--full-auto",
        "-C",
        str(worktree_path),
    ]
    if model:
        command.extend(["--model", model])
    values = {
        "prompt": prompt,
        "worktree": str(worktree_path),
        "log_path": str(log_path),
        "name": agent["name"],
        "model": agent.get("model", ""),
    }
    command.extend(format_parts(validate_string_list("harness_args", agent.get("harness_args")), values))
    command.append(prompt)
    return command


def build_custom_command(agent: dict, prompt: str, worktree_path: Path, log_path: Path) -> list[str]:
    command_template = validate_string_list("command", agent.get("command"), required=True)
    values = {
        "prompt": prompt,
        "worktree": str(worktree_path),
        "log_path": str(log_path),
        "name": agent["name"],
        "model": agent.get("model", ""),
    }
    return format_parts(command_template, values)


def build_agent_command(agent: dict, prompt: str, worktree_path: Path, log_path: Path) -> list[str]:
    harness = get_agent_harness(agent)
    if harness == "copilot":
        return build_copilot_command(agent, prompt, worktree_path, log_path)
    if harness == "codex":
        return build_codex_command(agent, prompt, worktree_path, log_path)
    if harness == "custom":
        return build_custom_command(agent, prompt, worktree_path, log_path)
    raise SystemExit(f"Unsupported harness {harness!r}. Use 'copilot', 'codex', or 'custom'.")


def build_window_command(agent: dict, worktree_path: Path, prompt: str, log_path: Path) -> str:
    agent_name = agent["name"]
    harness = get_agent_harness(agent)
    model = get_agent_model_display(agent)
    harness_cmd = shell_join(build_agent_command(agent, prompt, worktree_path, log_path))
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {shell_quote(str(worktree_path))}
        mkdir -p {shell_quote(str(log_path.parent))}
        echo "agent: {agent_name}"
        echo "harness: {harness}"
        echo "model: {model}"
        echo "worktree: {worktree_path}"
        echo "log: {log_path}"
        echo
        {harness_cmd} 2>&1 | tee -a {shell_quote(str(log_path))}
        status=${{PIPESTATUS[0]}}
        echo
        echo "agent exited with status $status"
        exec bash
        """
    ).strip()


def create_tmux_session(
    session_name: str,
    overview_text: str,
    windows: list[tuple[str, str]],
    replace: bool,
) -> None:
    if tmux_session_exists(session_name):
        if not replace:
            raise SystemExit(f"tmux session {session_name!r} already exists. Use --replace-session or rename it.")
        run(["tmux", "kill-session", "-t", session_name])

    overview_cmd = f"bash -lc {shell_quote(f'printf %s {shell_quote(overview_text)}; exec bash')}"
    run(["tmux", "new-session", "-d", "-s", session_name, "-n", "overview", overview_cmd], check=True)
    run(["tmux", "set-option", "-t", session_name, "mouse", "on"], check=True)
    run(["tmux", "set-option", "-t", session_name, "remain-on-exit", "on"], check=True)

    for window_name, command in windows:
        run(
            ["tmux", "new-window", "-t", session_name, "-n", window_name, f"bash -lc {shell_quote(command)}"],
            check=True,
        )


def build_overview_text(session_name: str, plans: list[dict], worktree_root: Path) -> str:
    lines = [
        f"tmux session: {session_name}",
        f"worktree root: {worktree_root}",
        "",
        "windows:",
    ]
    for plan in plans:
        lines.append(
            f"- {plan['name']}: harness={plan['harness']} model={plan['model']} branch={plan['branch']} worktree={plan['worktree']}"
        )
    lines.extend(
        [
            "",
            "tips:",
            "- switch windows: Ctrl+b n / Ctrl+b p",
            f"- attach later: tmux attach -t {session_name}",
            f"- kill session: tmux kill-session -t {session_name}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch parallel autoresearch agents in tmux.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without changing anything.")
    parser.add_argument("--attach", action="store_true", help="Attach to tmux after launch.")
    parser.add_argument(
        "--replace-session",
        action="store_true",
        help="Replace an existing tmux session with the same name.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    repo_root = Path(git(config_path.parent, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    run_tag = config["run_tag"]
    session_name = config.get("session_name", f"autoresearch-{run_tag}")
    base_ref = config.get("base_ref", "HEAD")
    worktree_root = resolve_worktree_root(repo_root, config)
    logs_root = worktree_root / "logs" / run_tag
    baseline = config["baseline"]
    agents = config["agents"]
    base_commit = git(repo_root, "rev-parse", "--short", base_ref).stdout.strip()

    if not agents:
        raise SystemExit("Config must define at least one agent.")

    dirty = git(repo_root, "status", "--short").stdout.strip()
    if dirty:
        print("warning: base checkout is dirty; new worktrees will start from committed HEAD only.", file=sys.stderr)

    plans: list[dict] = []
    for agent in agents:
        name = agent["name"]
        branch = agent.get("branch", f"autoresearch/{run_tag}-{name}")
        worktree = worktree_root / run_tag / name
        log_path = logs_root / f"{name}.log"
        prompt = build_prompt(agent, baseline)
        plans.append(
            {
                "name": name,
                "harness": get_agent_harness(agent),
                "model": get_agent_model_display(agent),
                "branch": branch,
                "worktree": str(worktree),
                "log_path": str(log_path),
                "prompt": prompt,
            }
        )

    if args.dry_run:
        print(f"repo_root={repo_root}")
        print(f"base_ref={base_ref}")
        print(f"base_commit={base_commit}")
        print(f"session_name={session_name}")
        print(f"worktree_root={worktree_root}")
        for plan in plans:
            print("---")
            print(f"agent={plan['name']}")
            print(f"harness={plan['harness']}")
            print(f"model={plan['model']}")
            print(f"branch={plan['branch']}")
            print(f"worktree={plan['worktree']}")
            print(f"log={plan['log_path']}")
        return 0

    windows: list[tuple[str, str]] = []
    for agent, plan in zip(agents, plans):
        worktree_path = Path(plan["worktree"])
        created, _ = ensure_worktree(repo_root, base_ref, plan["branch"], worktree_path)
        if created:
            print(f"created worktree {worktree_path}")
        baseline_added = ensure_baseline_commit(repo_root, worktree_path, base_commit, baseline, plan["name"])
        if baseline_added:
            print(f"recorded baseline in {worktree_path / 'results.tsv'}")
        window_command = build_window_command(agent, worktree_path, plan["prompt"], Path(plan["log_path"]))
        windows.append((plan["name"], window_command))

    overview_text = build_overview_text(session_name, plans, worktree_root)
    create_tmux_session(session_name, overview_text, windows, replace=args.replace_session)

    print(f"launched tmux session {session_name}")
    print(f"attach with: tmux attach -t {session_name}")
    if args.attach:
        subprocess.run(["tmux", "attach", "-t", session_name], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
