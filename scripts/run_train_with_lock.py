#!/usr/bin/env python3
"""
Serialize train.py runs across multiple agents sharing one machine.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LOCK_PATH = (Path.home() / ".cache" / "autoresearch" / "train.lock").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train.py with a shared machine-wide lock.")
    parser.add_argument(
        "--lock-path",
        default=str(DEFAULT_LOCK_PATH),
        help="Lock file path shared by all worktrees on this machine.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Seconds between lock/process checks while waiting.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=30.0,
        help="Seconds between status messages while waiting.",
    )
    return parser.parse_args()


def validate_lock_path(lock_path: Path) -> None:
    if lock_path == DEFAULT_LOCK_PATH:
        return
    if os.environ.get("AUTORESEARCH_ALLOW_CUSTOM_LOCK_PATH") == "1":
        return
    raise SystemExit(
        "Custom lock paths are disabled by default so agents cannot bypass shared training serialization. "
        "Use the default lock path, or set AUTORESEARCH_ALLOW_CUSTOM_LOCK_PATH=1 if you really need an override."
    )


def write_lock_info(info_path: Path, info: dict) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def read_lock_info(info_path: Path) -> dict | None:
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return None


def get_process_cwd(pid: int) -> str | None:
    proc = subprocess.run(
        ["lsof", "-a", "-d", "cwd", "-p", str(pid), "-Fn"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


def is_train_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False

    token_names = [Path(token).name for token in tokens]
    if "run_train_with_lock.py" in token_names:
        return False
    if "train.py" not in token_names:
        return False

    launcher_names = {"bash", "sh", "zsh", "node", "copilot"}
    if Path(tokens[0]).name in launcher_names:
        return False
    return True


def find_other_train_processes(repo_root: Path) -> list[tuple[int, str, str]]:
    current_pid = os.getpid()
    parent_pid = os.getppid()
    repo_marker = repo_root.name
    proc = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        text=True,
        capture_output=True,
        check=True,
    )

    matches: list[tuple[int, str, str]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid = int(parts[0])
        command = parts[1]
        if pid in {current_pid, parent_pid}:
            continue
        if not is_train_command(command):
            continue
        cwd = get_process_cwd(pid)
        if not cwd or repo_marker not in cwd:
            continue
        matches.append((pid, cwd, command))
    return matches


def wait_for_lock(lock_path: Path, info_path: Path, cwd: Path, poll_seconds: float, status_interval: float):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    last_status = 0.0

    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            now = time.time()
            if now - last_status >= status_interval:
                info = read_lock_info(info_path)
                if info:
                    print(
                        "Waiting for training lock held by "
                        f"pid={info.get('pid')} cwd={info.get('cwd')} started_at={info.get('started_at')}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print("Waiting for training lock...", file=sys.stderr, flush=True)
                last_status = now
            time.sleep(poll_seconds)


def wait_for_other_training(repo_root: Path, poll_seconds: float, status_interval: float) -> None:
    last_status = 0.0
    while True:
        matches = find_other_train_processes(repo_root)
        if not matches:
            return
        now = time.time()
        if now - last_status >= status_interval:
            summary = ", ".join(f"{pid}@{cwd}:{command[:80]}" for pid, cwd, command in matches)
            print(
                f"Lock acquired, but other train.py processes are still active; waiting: {summary}",
                file=sys.stderr,
                flush=True,
            )
            last_status = now
        time.sleep(poll_seconds)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    lock_path = Path(args.lock_path).expanduser().resolve()
    validate_lock_path(lock_path)
    info_path = lock_path.with_suffix(lock_path.suffix + ".json")
    cwd = Path.cwd().resolve()

    handle = wait_for_lock(lock_path, info_path, cwd, args.poll_seconds, args.status_interval)
    info = {
        "pid": os.getpid(),
        "cwd": str(cwd),
        "repo_root": str(repo_root),
        "hostname": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "state": "holding_lock",
    }
    write_lock_info(info_path, info)

    try:
        wait_for_other_training(repo_root, args.poll_seconds, args.status_interval)
        info["state"] = "running_train"
        write_lock_info(info_path, info)
        print(f"Acquired training lock: {lock_path}", file=sys.stderr, flush=True)
        process = subprocess.Popen([sys.executable, "train.py"], cwd=repo_root)
        return process.wait()
    finally:
        try:
            if info_path.exists():
                info_path.unlink()
        except OSError:
            pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
