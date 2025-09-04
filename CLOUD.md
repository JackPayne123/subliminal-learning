## Serverless GPU Job (RunPod or similar)

This repo includes a ready-to-use workflow for running `extensions/three_brains_experiment.py` on a pay-per-job GPU container.

### Files
- `cloud/runpod-job.sh`: container entrypoint script. Installs deps with uv, runs the experiment, optionally uploads results to the Hugging Face Hub.
- `cloud/requirements-runpod.txt`: minimal runtime dependencies for the container.
- `scripts/submit_runpod_job.py`: local client to submit RunPod serverless jobs.

### Environment Variables
Set these locally and/or in the container:
- `HF_TOKEN`: Hugging Face token (optional, required for private models or uploading).
- `HF_USER_ID`: HF username/namespace used by some workflows.
- `HF_UPLOAD_REPO`: Optional target dataset repo to upload results, e.g. `yourname/three-brains-results`.
- `RUNPOD_API_KEY`: Required by `scripts/submit_runpod_job.py`.

### Container Template
Create a RunPod Serverless template using a CUDA-enabled image (e.g., `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).

Your template should:
1. Clone this repo and checkout the requested commit/branch.
2. Execute `bash cloud/runpod-job.sh`.

Example startup command inside the template (pseudo):

```bash
git clone "$INPUT_REPO_URL" repo && cd repo && git checkout "$INPUT_COMMIT"
${INPUT_COMMAND}
```

### Submit a Job

```bash
export RUNPOD_API_KEY=...  # required

python scripts/submit_runpod_job.py \
  --template-id rpct-xxxxxxxx \
  --repo-url https://github.com/your/repo.git \
  --commit main \
  --env HF_TOKEN=$HF_TOKEN HF_USER_ID=$HF_USER_ID HF_UPLOAD_REPO=yourname/three-brains-results \
  --model-id unsloth/Qwen2.5-7B-Instruct \
  --dataset-size 2000 \
  --batch-size 8 \
  --max-length 768 \
  --save-dir /workspace/output/three_brains
```

The job will run `cloud/runpod-job.sh` inside the container with the provided environment. Results are saved to `SAVE_DIR`. If `HF_TOKEN`, `HF_USER_ID`, and `HF_UPLOAD_REPO` are set, the directory is uploaded to the Hugging Face Hub as a dataset.

### Vast.ai
This same script can be used on Vast.ai by launching an instance, SSH-ing into the container, cloning this repo, and running:

```bash
bash cloud/runpod-job.sh
```

Set the same environment variables as above.


