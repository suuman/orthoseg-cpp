You are working with two existing repositories:

MONAI backend:

```
/run/media/suman/Data/monai
```

Existing OrthoSeg UI:

```
/home/suman/agentic_coding/orthoseg
```

The BASIC MONAI integration is already implemented or should remain independent:

```
X-ray
  ↓
AI Segment
  ↓
MONAI prediction
  ↓
user edits
  ↓
Save / Export
  ↓
optionally Add to AI Training
  ↓
corrected image/mask added to training pool
```

IMPORTANT:

DO NOT change this normal annotation workflow.

DO NOT automatically start fine-tuning after Save/Export.

DO NOT start training simply because a corrected mask was added to the training pool.

This task adds a SEPARATE OPTIONAL model-management / training-control interface intended for an administrator, developer, or authorized technical user.

The normal annotator should still only see:

```
AI Segment
editing tools
Save / Export
Add to AI Training
```

Fine-tuning and model promotion remain deliberate/manual operations.

---

# 1. First inspect the MONAI backend

Start in:

```
/run/media/suman/Data/monai
```

Determine what ALREADY exists.

Inspect:

* server/API
* configuration
* training code
* training status
* dataset counters
* `finetune.sh`
* candidate model handling
* validation
* production model handling
* candidate promotion
* archive/rollback support
* model reload
* logging
* training logs
* metrics
* model metadata

Determine whether endpoints already exist approximately like:

```
GET /health
GET /model/info
GET /training/status
POST /training/start
GET /training/jobs/{id}
POST /model/promote
POST /model/reload
```

DO NOT assume these exist.

Use the REAL implementation.

If training is currently intentionally exposed only through scripts such as:

```
./scripts/finetune.sh
```

and:

```
./scripts/promote_candidate.sh
```

preserve that design unless a very small backend API layer is required for this management interface.

Do NOT rewrite the training system.

---

# 2. Then inspect OrthoSeg

Inspect:

```
/home/suman/agentic_coding/orthoseg
```

Determine:

* UI framework
* menus/toolbars
* settings/admin panels
* dialogs
* networking abstraction
* threading/background job infrastructure
* status/progress components
* table/list widgets
* logging mechanisms
* existing MONAI client
* existing MONAI integration

Reuse all existing patterns.

Make minimal changes.

---

# 3. Keep model management separate from annotation

Do NOT place training controls next to:

```
Save
Export
Add to AI Training
```

Do NOT add:

```
Train Now
```

to the normal save dialog.

Instead create a separate location such as:

```
Tools
    → AI Model Management
```

or:

```
Settings
    → AI Model Management
```

or another existing advanced/admin area.

Use the UI's established conventions.

This panel is for technical users.

---

# 4. Suggested Model Management panel

Add an optional panel/dialog approximately like:

```
AI Model Management
───────────────────────────────────────

Backend
Status:              ● Ready
Device:              NVIDIA GPU
Backend URL:         http://127.0.0.1:8000

Production Model
Version:             femur_tibia_2d_v003
Architecture:        UNet
Training Cases:      527
Created:             ...
Mean Val Dice:       ...

Training Dataset
Approved Cases:      545
New Since Training:  18
Validation Cases:    75

Latest Candidate
Version:             femur_tibia_2d_v004
Status:              Ready
Mean Dice:           0.9824
Femur Dice:          0.9862
Tibia Dice:          0.9786

[Fine-tune New Candidate]

[Refresh]

[Promote Candidate]
```

Do not copy this layout blindly if the existing UI has a better established pattern.

Keep it simple.

---

# 5. Backend information

The panel should display only information available from the actual backend.

At minimum, when available:

## Backend

* online/offline
* device
* GPU name
* MONAI/backend health
* loaded model status

## Production model

* model version
* architecture
* training date
* training case count
* validation metrics
* parent version if available

## Dataset

* total approved training cases
* number of new cases since previous training
* validation case count

## Candidate

* candidate version
* training status
* validation metrics
* parent production model

Do not invent missing metadata.

Display:

```
N/A
```

if the backend genuinely does not provide it.

---

# 6. Refresh

Provide:

```
Refresh
```

which reloads model/training information from the backend.

Do not poll aggressively.

A refresh when the panel opens plus manual refresh is sufficient unless an active training job is running.

---

# 7. Fine-tune control

Add:

```
Fine-tune New Candidate
```

This action is MANUAL.

It must never be triggered by:

* Save
* Export
* Add to AI Training
* application startup
* closing a case
* reaching a certain number of masks automatically

When selected, show a confirmation dialog.

For example:

```
Start fine-tuning a new candidate model?

Production model:
    femur_tibia_2d_v003

Approved training cases:
    545

New cases since last training:
    18

Validation cases:
    75

The current production model will NOT
be replaced automatically.

[Cancel]
[Start Fine-tuning]
```

Do not imply the candidate will automatically become production.

---

# 8. Training configuration

Prefer using backend defaults.

Do not expose dozens of ML hyperparameters to normal users.

Optionally expose only a small Advanced section if the backend supports them safely:

```
Epochs
Batch size
Learning rate
New-case repeat factor
Resume from production
```

But DEFAULT behavior should be:

```
use backend configuration
```

For example:

```
Fine-tuning Configuration:
● Use backend defaults
○ Advanced
```

Do not duplicate the backend configuration system in the UI.

---

# 9. Starting training

Prefer an existing backend training endpoint if one already exists.

If the backend supports something approximately like:

```
POST /training/start
```

use it.

If the existing backend intentionally supports training only via:

```
./scripts/finetune.sh
```

then determine the cleanest safe integration.

Preferred options, in order:

1. Existing backend training API.
2. Add a very thin localhost-only backend endpoint which launches the EXISTING training mechanism.
3. Only if appropriate to the current architecture, execute the existing script through a controlled local process.

Do NOT duplicate the Python training implementation in OrthoSeg.

Do NOT run MONAI/PyTorch directly from UI code.

---

# 10. Security of training controls

Training/model-management endpoints should remain local.

Do not expose them publicly.

Default backend:

```
127.0.0.1
```

If new admin endpoints are needed:

* keep them localhost-only
* reuse backend validation
* do not allow arbitrary shell command execution
* do not accept arbitrary filesystem paths from UI
* do not expose generic command execution

For example, DO NOT implement:

```
POST /run-command
command="..."
```

Instead expose explicit safe operations such as:

```
POST /training/start
```

or:

```
POST /models/{version}/promote
```

---

# 11. Training must run asynchronously

Training can take significant time.

DO NOT block the OrthoSeg UI while training.

Expected behavior:

```
Start Fine-tuning
      ↓
backend starts job
      ↓
UI receives job ID/status
      ↓
user continues using UI
      ↓
management panel shows progress
```

The annotation application should remain usable.

---

# 12. Training status

If supported by backend, show:

```
Training Status
─────────────────────────

Candidate:
    femur_tibia_2d_v004

Status:
    Training

Epoch:
    17 / 50

Best validation Dice:
    0.9784

Started:
    ...
```

Possible states:

```
Idle
Queued
Preparing
Training
Validating
Completed
Failed
Cancelled
```

Use the backend's actual states if different.

---

# 13. Training logs

If backend provides logs, optionally show a small read-only log area:

```
Training Log
```

Do not dump massive logs into the UI indefinitely.

Show perhaps:

* recent lines
* progress
* errors
* current epoch
* current validation score

Provide:

```
View Full Log
```

only if appropriate.

Do not reimplement terminal emulation.

---

# 14. Failure handling

If training fails:

show:

```
Fine-tuning failed.
```

and display the useful backend error.

Do NOT:

* replace production model
* remove production checkpoint
* remove current candidate history
* corrupt training data

Current production inference must remain unaffected.

---

# 15. Candidate creation

Training must create a CANDIDATE model.

For example:

```
production:
    femur_tibia_2d_v003
```

Fine-tuning creates:

```
candidate:
    femur_tibia_2d_v004
```

Do NOT automatically replace:

```
production/model.pt
```

after successful training.

---

# 16. Candidate metrics

After training completes, display candidate validation metrics available from the backend.

At minimum when supported:

```
Mean foreground Dice
Femur Dice
Tibia Dice
```

Optionally:

```
IoU
HD95
```

Also show production metrics alongside candidate metrics if available.

Example:

```
Validation Metrics

                     Production      Candidate
Femur Dice             0.9821          0.9862
Tibia Dice             0.9742          0.9786
Mean Dice              0.9782          0.9824
```

This is informational.

Do not automatically decide that the candidate is "better" based solely on one metric unless backend acceptance logic explicitly provides that decision.

---

# 17. Promotion must be manual

Provide:

```
Promote Candidate
```

ONLY when a valid candidate exists.

When clicked, show explicit confirmation:

```
Promote candidate model?

Current production:
    femur_tibia_2d_v003

New candidate:
    femur_tibia_2d_v004

The current production model will be archived.

[Cancel]
[Promote]
```

Promotion must remain an intentional action.

---

# 18. Use existing promotion mechanism

Inspect the backend.

If it already uses:

```
./scripts/promote_candidate.sh <version>
```

reuse the existing promotion implementation.

Do NOT reimplement checkpoint copying/versioning independently in OrthoSeg.

Preferred:

```
UI
  ↓
explicit backend promotion API
  ↓
existing backend promotion logic
```

If an endpoint is needed, make it a very thin wrapper around existing safe promotion functionality.

---

# 19. Production model archive

Promotion should preserve the old production model.

Do not delete old production checkpoints.

Expected backend organization may be approximately:

```
models/
  production/
  candidates/
  archive/
```

Use the backend's existing structure.

UI should simply report successful promotion.

---

# 20. Reload after promotion

Determine how the backend currently reloads a production checkpoint.

It may require:

```
server restart
```

or support:

```
POST /model/reload
```

If safe reload already exists:

use it.

If restart is required:

display:

```
Candidate promoted successfully.
Restart the MONAI backend to load the new model.
```

Do not introduce complicated hot-reloading unless necessary.

---

# 21. Show currently loaded model

This is important.

Differentiate:

```
Production model on disk
```

from:

```
Model currently loaded for inference
```

if those can differ.

For example:

```
Production:
    femur_tibia_2d_v004

Loaded:
    femur_tibia_2d_v003

Restart/reload required.
```

This avoids confusion.

---

# 22. Optional model history

If backend metadata already supports it, add a simple:

```
Model History
```

view.

For example:

```
v004    Candidate    2026-09-22
v003    Production   2026-09-10
v002    Archived     2026-08-31
v001    Archived     2026-08-15
```

Do not build a database solely for this if backend metadata/files already provide the information.

---

# 23. Optional rollback

Only add rollback if the backend already supports safe model versioning/archive.

Possible UI:

```
Select archived model
[Restore as Production]
```

Require confirmation.

Do NOT implement unsafe arbitrary checkpoint selection.

Only allow valid known archived model versions.

Rollback should use backend logic, not direct arbitrary file manipulation from OrthoSeg.

---

# 24. Never mix candidate training with annotation data submission

Keep these operations conceptually distinct:

## Annotation action

```
Add to AI Training
```

means:

```
store corrected image/mask
```

It does NOT:

```
train
```

## Administrative action

```
Fine-tune New Candidate
```

means:

```
train using approved training pool
```

This separation is essential.

---

# 25. No automatic trigger based on training-case count

The panel may display:

```
New cases since training: 20
```

but DO NOT automatically start fine-tuning when it reaches 20.

It is acceptable to display:

```
20 new approved cases available.
```

But training still requires:

```
Fine-tune New Candidate
```

from an authorized user.

---

# 26. Keep AI Segment independent

Normal AI Segment must always use the currently loaded production model.

Do not make it use:

* partially trained models
* incomplete candidate checkpoints
* arbitrary latest checkpoint

Only the production/loaded model should serve normal inference.

---

# 27. Prevent training/inference conflicts where necessary

Inspect how the backend uses the GPU.

If the workstation cannot safely perform annotation inference and training simultaneously, handle this cleanly.

Possible backend policy:

```
training active
    ↓
inference temporarily unavailable
```

or:

```
inference allowed with resource control
```

Use the existing/backend-appropriate approach.

Do not guess.

If inference is unavailable during training, OrthoSeg should display:

```
MONAI training is currently running.
AI Segment is temporarily unavailable.
```

Manual annotation and MedSAM2 should remain functional where technically possible.

---

# 28. Training-data summary

The Model Management panel may show:

```
Training Data
─────────────────

Approved cases:           545
Used by production:       527
New approved cases:        18
Validation cases:          75
```

Only show values available from backend metadata.

Do not count files independently in OrthoSeg if the backend already owns dataset accounting.

---

# 29. Optional "View Training Cases"

Do NOT add a complex dataset editor unless explicitly required.

If useful, a simple read-only summary can be provided.

Do not allow model-management UI to arbitrarily delete training cases unless backend already has an established safe workflow.

This task is primarily model management, not dataset curation.

---

# 30. Do not expose sensitive implementation details unnecessarily

The UI may display:

```
model version
training metrics
status
```

It does not need to display absolute paths such as:

```
/run/media/suman/Data/monai/...
```

to ordinary users.

Keep filesystem implementation details in logs/admin diagnostics if necessary.

---

# 31. Keep UI changes modest

Prefer a single:

```
AI Model Management
```

dialog/panel.

Do NOT add model controls throughout the application.

Do NOT clutter the normal annotation toolbar.

The normal user interface should remain focused on annotation.

---

# 32. Suggested final panel

Conceptually:

```
┌─────────────────────────────────────────┐
│ AI Model Management                     │
├─────────────────────────────────────────┤
│ Backend                                 │
│ Status: Ready                           │
│ GPU: NVIDIA ...                         │
│                                         │
│ Production Model                        │
│ Version: femur_tibia_2d_v003            │
│ Mean Dice: 0.9782                       │
│                                         │
│ Training Data                           │
│ Approved: 545                           │
│ New since training: 18                  │
│ Validation: 75                          │
│                                         │
│ [Fine-tune New Candidate]               │
│                                         │
│ Candidate                               │
│ Version: femur_tibia_2d_v004            │
│ Status: Completed                       │
│ Mean Dice: 0.9824                       │
│                                         │
│ [Promote Candidate]                     │
│                                         │
│ [Refresh]                               │
└─────────────────────────────────────────┘
```

Adapt to the existing framework and visual conventions.

Do not redesign the overall application.

---

# 33. Tests

Add focused tests only.

At minimum:

## Status

* backend status loads
* offline backend handled
* production metadata displayed
* dataset counters displayed

## Training

* training starts only through explicit user action
* annotation submission does NOT start training
* duplicate train-click prevented while job already active
* UI remains responsive
* failed training does not affect production model

## Candidate

* candidate information displayed
* metrics displayed correctly
* invalid/missing candidate disables Promote

## Promotion

* explicit confirmation required
* correct candidate version sent
* failed promotion leaves current production unchanged
* successful promotion refreshes model metadata

## Regression

Confirm existing:

* AI Segment
* AI Fill / MedSAM2
* annotation editing
* Save
* Export
* Add to AI Training

still work.

---

# 34. Acceptance criteria

Complete only when:

1. Normal annotation workflow is unchanged.
2. Save/Export still only optionally adds training data.
3. Adding training data NEVER starts training.
4. Model management is in a separate advanced/admin area.
5. Production model information is visible.
6. Training dataset status is visible.
7. User can manually start fine-tuning.
8. Training runs asynchronously.
9. Production model remains active during/after training unless explicitly promoted.
10. Candidate metrics can be inspected.
11. Candidate promotion requires explicit confirmation.
12. Existing backend model-versioning logic is reused.
13. Previous production model is preserved according to backend logic.
14. Model reload/restart behavior is clear.
15. Existing MedSAM2/AI Fill is unaffected.
16. No training implementation is duplicated in OrthoSeg.
17. MONAI backend remains the authority for training and model management.

---

# 35. Final implementation report

At completion provide:

1. Existing backend training/model APIs discovered.
2. Any backend APIs added, if required.
3. Why each backend change was necessary.
4. OrthoSeg files modified.
5. Location of AI Model Management in the UI.
6. How production-model information is obtained.
7. How manual fine-tuning is started.
8. How training status/progress is obtained.
9. How candidate metrics are displayed.
10. How promotion works.
11. How model reload/restart works.
12. Confirmation that Add to AI Training does NOT start training.
13. Confirmation that normal annotation workflow is unchanged.
14. Confirmation that MedSAM2 was untouched.
15. Test/build results.
16. Exact commands to run backend and UI.

IMPORTANT:

This is an OPTIONAL ADMINISTRATIVE MODEL-MANAGEMENT FEATURE.

Do not couple it tightly to normal annotation.

Do not automatically fine-tune.

Do not automatically promote candidates.

Do not perform unrelated refactoring.

Reuse the existing MONAI backend's training, validation, checkpoint, and promotion implementation wherever possible.

