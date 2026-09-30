# Remote MONAI server and local OrthoSeg implementation plan

Saved: 2026-09-30

## Objective and status

Run MONAI inference and fine-tuning on a remote GPU server while running OrthoSeg locally for image viewing, annotation, and mask editing. Keep corrected cases, training jobs, and model versions synchronized through the backend API.

This document records recommendations from the repository review. The deployment changes below are proposed; they have not yet been implemented or deployed.

## Recommended architecture

```text
┌──────────────────────────────────────┐
│          LOCAL COMPUTER              │
│                                      │
│  ┌────────────────────────────────┐  │
│  │          OrthoSeg UI           │  │
│  │ Upload · Edit · Export masks   │  │
│  │ Model management · Fine-tuning │  │
│  └───────────────┬────────────────┘  │
│                  │                   │
│  ┌───────────────▼────────────────┐  │
│  │ Local images and masks         │  │
│  │ Pending submission queue*      │  │
│  └───────────────┬────────────────┘  │
│                  │                   │
│  ┌───────────────▼────────────────┐  │
│  │        MONAI API client        │  │
│  └───────────────┬────────────────┘  │
└──────────────────┼───────────────────┘
                   │
          Encrypted SSH tunnel
                   │
       Images, masks, job requests ↑↓
       Predictions, status, versions
                   │
┌──────────────────▼───────────────────┐
│          REMOTE GPU SERVER           │
│                                      │
│  ┌────────────────────────────────┐  │
│  │           MONAI API            │  │
│  └───────┬───────────────┬────────┘  │
│          │               │           │
│  ┌───────▼──────┐ ┌──────▼────────┐  │
│  │ GPU inference│ │ Training job  │  │
│  │ service      │ │ manager      │  │
│  └───────▲──────┘ └──────┬────────┘  │
│          │               │           │
│          │        ┌──────▼────────┐  │
│          │        │ GPU fine-     │  │
│          │        │ tuning worker*│  │
│          │        └──────┬────────┘  │
│          │               │           │
│  ┌───────┴───────────────▼────────┐  │
│  │ Versioned model registry       │  │
│  │ Candidates → Evaluate → Promote│  │
│  └────────────────────────────────┘  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ Persistent server storage      │  │
│  │ Images · Labels · Revisions    │  │
│  │ Training snapshots · Job logs  │  │
│  │ Model checkpoints · Backups    │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

\* Proposed changes: persistent upload queue and GPU support for UI-started fine-tuning. Server storage supplies training datasets; promoted models require activation before inference uses them. The queue applies to approved training submissions; inference and management requests go directly from the UI to the API client.

Use the remote server as the authority for submitted cases, revisions, jobs, and models. Synchronize through APIs rather than bidirectionally copying mutable data folders. Local files remain available for editing and export; a successful server acknowledgement determines whether a training submission is synchronized.

## What currently works and what needs changing

| Area | Current behavior | Recommended change |
| --- | --- | --- |
| Backend connection | OrthoSeg accepts HTTP loopback URLs only | Start with SSH local forwarding; no relaxation of the URL restrictions is needed |
| Management access | Enabled explicitly and restricted to loopback requests | Keep the backend bound to loopback and use the tunnel |
| UI-started fine-tuning | Job manager forces the device to CPU | Add independent training device configuration and remove the forced CPU override |
| CLI fine-tuning | Uses the existing inference device configuration | GPU training is available with `inference.device: cuda`; separate the settings in the implementation |
| Model portability | Checkpoints can retain machine-specific asset paths | Resolve runtime assets from the server configuration |
| Submission synchronization | Uploads exist, but no durable client queue or idempotent submission protocol | Add persistent submissions, retry handling, acknowledgements, and revision reconciliation |
| Model promotion | Changes the on-disk model link; loaded model changes after restart | Expose loaded/promoted versions and implement a controlled activation workflow |
| GPU memory | Startup attempts to load UNet, SAM2, and nnUNet | Add model enablement/lazy loading and a training/inference GPU scheduling policy |
| Local ONNX models | Independent of remote PyTorch model promotion | Use remote inference for immediate consistency, or add a separate ONNX distribution workflow |

## 1. Connect local OrthoSeg to the remote server

`orthoseg/src/MonaiClient.cpp` reads `MONAI_BACKEND_URL`, defaulting to `http://127.0.0.1:8000`. It currently accepts only HTTP URLs whose host is `127.0.0.1`, `localhost`, or `::1`, and rejects credentials, query strings, and fragments. The client also disables HTTP proxy use.

An SSH local forward fits these existing restrictions:

```bash
ssh -NT \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  user@gpu-server
```

Start OrthoSeg locally with:

```bash
MONAI_BACKEND_URL=http://127.0.0.1:18000 \
  /home/suman/agentic_coding/orthoseg/build/orthoseg
```

Replace `user@gpu-server` with the actual SSH account and hostname. Keep the remote MONAI listener at `127.0.0.1:8000`. Supervise the tunnel for routine use so dropped connections can recover.

The current management gate in `app/api/management.py` requires management to be enabled, a loopback client address and Host hostname, and no Origin header. Native Qt requests through the tunnel fit this arrangement. The API otherwise has no authentication; do not expose it directly to the public network. A future direct HTTPS connection would require authentication, authorization, TLS, and corresponding client/server changes.

References: [SSH local forwarding](https://man.openbsd.org/ssh.1), [SSH keepalive and forwarding options](https://man.openbsd.org/ssh_config.5).

## 2. Enable GPU fine-tuning from Model Management

The most important functional blocker is in `app/management/jobs.py`: it copies the configuration and sets `config["inference"]["device"] = "cpu"` before launching training. The management summary also reports a CPU training policy.

`app/ml/model.py::get_device` currently reads `inference.device` for device selection and supports `auto`, `cpu`, and `cuda`. It does not currently support `cuda:N`.

Required changes:

1. Introduce `training.device` independently of `inference.device`.
2. Update defaults and strict configuration validation in `app/core/config.py`; unknown configuration keys are currently rejected.
3. Remove the job manager's unconditional CPU override and make both UI jobs and CLI training use the training device setting.
4. Report the actual selected device, CUDA availability, and useful GPU status in management responses and the UI.
5. Define GPU allocation and concurrency behavior. On one GPU, pause or queue competing work when inference plus training would exceed available memory. On multiple GPUs, support explicit allocation if needed.
6. Keep CPU execution available and make unavailable requested GPU configurations produce a clear error.

`training.device` and explicit GPU allocation are proposed settings, not settings supported by the current code.

### Existing CLI fallback

Fine-tuning can already be started through `scripts/finetune.sh` using a server configuration with `inference.device: cuda`:

```bash
cd /srv/monai
XRAY_PYTHON=/srv/monai/.venv/bin/python \
  ./scripts/finetune.sh \
  --config configs/remote.yaml \
  --model medsam2 \
  --epochs 50
```

`/srv/monai` and `configs/remote.yaml` are example deployment paths; create the server configuration before running this command. UI-started training will still use CPU until the job manager changes above are implemented.

## 3. Make checkpoints portable between machines

`app/ml/checkpoint.py::load_checkpoint` builds the model from configuration stored in checkpoint metadata. For MedSAM2, `app/ml/sam2_model.py::asset_root` resolves assets using the stored `paths.models` plus `sam2.asset_directory`.

A checkpoint created on the workstation may therefore refer to an absolute workstation path that does not exist on the server.

Required changes:

- Keep model architecture and preprocessing settings associated with the checkpoint.
- Resolve asset locations from the active server configuration rather than stale checkpoint filesystem paths.
- Validate required assets before launching inference or training, with actionable errors.
- Verify a checkpoint created under one root can load under a different root without depending on the old directory.

Deploy the necessary assets along with the backend, including the configured MedSAM2 package/configuration/checkpoint and nnUNet plans, dataset metadata, and fold checkpoint where those models are enabled.

## 4. Add reliable case and annotation synchronization

The current client submits the original PNG, mask, original filename, model version, and a case identifier based on the original bytes. The backend also deduplicates by decoded pixels, so re-encoded versions of the same image can resolve to the same case. Each accepted repeat submission still creates a new UUID revision.

There is currently no persistent upload queue, submission idempotency key, or optimistic base-revision conflict handling.

Required changes:

1. Save approved submissions in a durable local queue, including an immutable image/mask snapshot and metadata, before uploading.
2. Assign a stable submission ID used across retries. The server must return the original acknowledgement for a repeated ID instead of creating another revision.
3. Return and persist the canonical server case ID and accepted revision ID.
4. Include the base revision when updating an existing annotation and detect conflicting edits instead of silently replacing them.
5. Retry transient failures across application restarts and tunnel outages; retain visible failures that require intervention.
6. Show synchronization state in Model Management: pending upload, uploading, synchronized, failed, or conflict.
7. Keep filename, labels, revision, and training incorporation status available in the existing case list and export.

Only explicitly approved corrections should enter the training submission queue. A locally exported mask alone should not imply approval to train.

## 5. Keep model versions consistent

Promotion currently updates a model symlink atomically on disk, but a running server continues to use its loaded model until restarted. Health/status already distinguishes the on-disk and loaded versions.

Required changes:

- Display both promoted and loaded versions so users can see whether activation is pending.
- Provide a controlled restart or model reload process that accounts for in-flight requests.
- Record the inference model version with predictions and subsequent approved corrections.
- Refresh health/model status after activation and reconnection.
- Retain prior versions for rollback.

Local ONNX models do not automatically change when the remote PyTorch model is promoted. Prefer MONAI-backed inference choices when users need the current remote model. If updated local inference is required, add a separate ONNX export, validation, download, checksum, and version-selection workflow.

## 6. Manage GPU memory and long-running requests

`app/main.py` currently attempts to load UNet, SAM2, and nnUNet predictors at startup. These models plus a training worker may exceed one GPU's memory.

Add enabled-model configuration or lazy loading, and specify how inference and training share the GPU. Keep training jobs asynchronous and use job polling for progress rather than holding a request open for the entire training run.

The OrthoSeg client defaults to a 180,000 ms timeout; health and management status use a shorter timeout capped at 5,000 ms. It uses both transfer timeouts and an absolute request timer. Remote connections need separately configurable upload, inference, and status deadlines so slow uploads and GPU queueing are handled deliberately.

## 7. Server installation and operations

- Deploy a versioned backend checkout and recreate its Python environment on the server; do not copy the workstation virtual environment.
- Install a PyTorch build appropriate to the server GPU/driver and verify `torch.cuda.is_available()` before enabling GPU jobs. See the [official PyTorch installation guidance](https://docs.pytorch.org/get-started/locally/).
- Configure persistent data and model directories with backups. Keep datasets, revisions, job records, checkpoints, and promotion metadata across application upgrades.
- Use a dedicated remote configuration with loopback binding, server-local paths, and management enabled when required.
- Run the backend under a service manager and supervise the client tunnel.
- Pin compatible client/backend versions and add an API protocol/capability handshake so incompatible upgrades fail clearly.
- Back up server state and test restoration, including model activation and case revision history.

`app/serve.py --config` accepts a configuration path and starts Uvicorn with one worker. Keep one worker until model loading and job coordination explicitly support more.

`scripts/common.sh` selects Python from `XRAY_PYTHON`, then `VIRTUAL_ENV`, the repository `.venv`, or `python3`. Some model server wrappers prefer `/home/suman/deepnet/bin/python` if present; explicitly setting `XRAY_PYTHON` makes the remote interpreter choice predictable.

## Implementation order

1. Separate inference/training device settings and enable GPU jobs from Model Management.
2. Make checkpoint asset resolution portable.
3. Add the durable upload queue, idempotent submissions, acknowledgements, and revision conflict handling.
4. Expose synchronization and loaded/promoted model status in OrthoSeg.
5. Add GPU resource policy, remote request settings, and deployment/service templates.
6. Validate end to end on the target server before routine use.

## Validation checklist

- Connect local OrthoSeg through the tunnel and verify health, inference, management status, and case uploads.
- Start fine-tuning from the UI and confirm the worker actually uses the configured GPU.
- Load checkpoints after moving the model directory to a different filesystem root.
- Disconnect during submission, restart OrthoSeg, reconnect, and verify exactly one accepted revision for the retried submission.
- Submit competing changes based on the same revision and verify visible conflict handling.
- Promote a candidate and verify the UI accurately reports the loaded version before and after activation.
- Exercise inference during training according to the selected GPU sharing policy.
- Verify case listing/export shows filenames, labels, revisions, and incorporation state correctly.
- Restore a backup and verify cases, jobs, and active model metadata remain consistent.

No remote GPU deployment or remote training validation was performed as part of the recommendation review.
