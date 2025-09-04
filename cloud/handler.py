#!/usr/bin/env python3
"""
RunPod Serverless handler that clones a repo at a given commit and runs the job script.

Expects input fields:
  - repo_url: git URL of repo to clone
  - commit: branch/tag/sha to checkout (default: main)
  - command: command to run inside the repo (default: bash cloud/runpod-job.sh)
  - env: dict of environment variables to export before running the command
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict

import runpod


def _run(cmd: str, cwd: str | None = None) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input", {}) or {}
    repo_url: str = inp.get("repo_url") or os.environ.get("INPUT_REPO_URL", "")
    if not repo_url:
        return {"status": "error", "message": "Missing repo_url"}
    commit: str = inp.get("commit") or os.environ.get("INPUT_COMMIT", "main")
    command: str = inp.get("command") or os.environ.get("INPUT_COMMAND", "bash cloud/runpod-job.sh")
    env_vars: Dict[str, str] = inp.get("env") or {}

    # Export requested environment variables
    for k, v in env_vars.items():
        os.environ[str(k)] = str(v)

    workdir = "/workspace/job"
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    # Clone repo and checkout commit
    rc = _run(f"git clone {repo_url} .", cwd=workdir)
    if rc != 0:
        return {"status": "error", "message": "git clone failed", "exit_code": rc}
    rc = _run(f"git checkout {commit}", cwd=workdir)
    if rc != 0:
        return {"status": "error", "message": "git checkout failed", "exit_code": rc}

    # Run the provided command
    rc = _run(command, cwd=workdir)
    status = "success" if rc == 0 else "error"
    return {"status": status, "exit_code": rc}


runpod.serverless.start({"handler": handler})


