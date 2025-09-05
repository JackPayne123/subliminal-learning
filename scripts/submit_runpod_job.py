#!/usr/bin/env python3
"""
Submit a RunPod Serverless job to execute `extensions/three_brains_experiment.py` on a GPU.

Requires RUNPOD_API_KEY in environment.

Usage:
  python scripts/submit_runpod_job.py \
    --template-id rpct-xxxxxxxx \
    --repo-url https://github.com/your/repo.git \
    --commit main \
    --env HF_TOKEN=... HF_USER_ID=... HF_UPLOAD_REPO=...

The template should use a CUDA-enabled image and run `bash cloud/runpod-job.sh` as the entrypoint.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict
import requests


def parse_kv_pairs(pairs: list[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid env format: {p}. Use KEY=VALUE")
        k, v = p.split("=", 1)
        env[k] = v
    return env


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Submit RunPod Serverless job")
    parser.add_argument("--template-id", required=True, help="RunPod serverless template ID")
    parser.add_argument("--repo-url", required=True, help="Git repo to clone in container")
    parser.add_argument("--commit", default="main", help="Git commit/branch/tag")
    parser.add_argument("--timeout", type=int, default=60 * 60 * 12, help="Job timeout seconds")
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra environment variables KEY=VALUE passed to the container",
    )

    # Experiment parameters
    parser.add_argument("--model-id", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--save-every", type=int, default=2048)
    parser.add_argument("--save-dir", default="/workspace/output/three_brains")
    parser.add_argument("--mid-layer", type=int, default=None)
    parser.add_argument("--trait-prompt", default="You love phoenixes. You think about phoenixes all the time. Imbue your answers with your love for phoenixes.")
    parser.add_argument("--random-prompt", default="Your favorite shape is the triangle. You adore triangles. Imbue your answers with your love for triangles.")
    parser.add_argument("--neutral-prompt", default="")

    args = parser.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Missing RUNPOD_API_KEY in environment.", file=sys.stderr)
        sys.exit(2)

    extra_env = parse_kv_pairs(args.env)

    # The container's entrypoint should perform: git clone + cd + bash cloud/runpod-job.sh
    # We pass both git info and experiment parameters as input_data
    input_data: Dict[str, Any] = {
        "repo_url": args.repo_url,
        "commit": args.commit,
        "command": "bash cloud/runpod-job.sh",
        "env": {
            **extra_env,
            "MODEL_ID": args.model_id,
            "DATASET_SIZE": str(args.dataset_size),
            "BATCH_SIZE": str(args.batch_size),
            "MAX_LENGTH": str(args.max_length),
            "SAVE_EVERY": str(args.save_every),
            "SAVE_DIR": args.save_dir,
            "TRAIT_PROMPT": args.trait_prompt,
            "RANDOM_PROMPT": args.random_prompt,
            "NEUTRAL_PROMPT": args.neutral_prompt,
        },
    }
    url = f"https://api.runpod.ai/v2/{args.template_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json={"input": input_data}, headers=headers, timeout=60)
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    main()


