The updated MedSAM2 model and inference code are located in:

`/run/media/suman/Data/medsam2`

The updated trained checkpoint is:

`/run/media/suman/Data/medsam2/checkpoint.pt`

The reference Python inference implementation is:

`/run/media/suman/Data/medsam2/medsam2_infer_custom.py`

The updated MedSAM2 model produces independent bone-structure channels using the following fixed encoding:

- Background: `RGB(0, 0, 0)`
- Femur, Label 1: `RGB(1, 0, 0)`
- Tibia, Label 2: `RGB(0, 2, 0)`
- Femur + Tibia overlap: `RGB(1, 2, 0)`

Femur and tibia may overlap in 2D X-ray projection. Do not collapse the output into a mutually exclusive single-channel `0/1/2` mask during inference, annotation editing, training-pool storage, fine-tuning preparation, or model output post-processing.

Existing repositories:

MONAI backend:

`/run/media/suman/Data/monai`

Existing OrthoSeg UI:

`/home/suman/agentic_coding/orthoseg`

Keep the MONAI backend and OrthoSeg UI as separate repositories. Do not create a new backend inside the UI repository.

## Objective

Integrate the updated MedSAM2 model into the existing MONAI backend and OrthoSeg UI, including the existing model fine-tuning workflow.

The updated workflow must support:

1. inference with the new checkpoint,
2. mask and bounding-box prompting,
3. editing femur and tibia independently,
4. preserving femur/tibia overlap,
5. saving corrected annotations,
6. adding corrected annotations to the existing training pool,
7. launching the existing MedSAM2 fine-tuning workflow from the OrthoSeg/MONAI system,
8. tracking fine-tuning status,
9. managing resulting model checkpoints,
10. selecting or activating a newly fine-tuned checkpoint using the existing model-management workflow.

Make the minimum required changes. Reuse existing implementation wherever possible.

## Step 1 — Inspect the existing implementations first

Before modifying anything, inspect:

- `/run/media/suman/Data/medsam2/medsam2_infer_custom.py`
- `/run/media/suman/Data/monai`
- `/home/suman/agentic_coding/orthoseg`

Specifically identify:

- existing MedSAM2 integration,
- model-loading code,
- model registry/model-management code,
- training/fine-tuning endpoints,
- training-pool storage,
- corrected-annotation workflow,
- bounding-box and painted-mask prompt handling,
- training job creation,
- training status/progress reporting,
- checkpoint naming and storage,
- active-model selection,
- UI controls related to training and model management.

Do not duplicate functionality that already exists.

## Step 2 — Integrate the updated checkpoint

Use:

`/run/media/suman/Data/medsam2/checkpoint.pt`

and use:

`/run/media/suman/Data/medsam2/medsam2_infer_custom.py`

as the behavioural reference for preprocessing, inference, prompts, and output post-processing.

Preserve independent masks internally:

```text
femur_mask: binary 0/1
tibia_mask: binary 0/1
```

The combined encoded mask must follow:

```text
Background           = RGB(0, 0, 0)
Femur only            = RGB(1, 0, 0)
Tibia only            = RGB(0, 2, 0)
Femur + Tibia overlap = RGB(1, 2, 0)
```

An overlap pixel is valid:

```text
femur = 1
tibia = 1
```

Do not use mutually exclusive class assignment between femur and tibia.

## Step 3 — Preserve existing prompt and refinement modes

Continue to support the existing OrthoSeg workflow for:

- bounding-box prompts,
- painted-mask prompts,
- masks loaded from files,
- femur selection,
- tibia selection,
- AI Fill/refinement.

Where the current AI Fill workflow already exists, update it rather than adding a duplicate implementation.

## Step 4 — Update annotation representation

Internally preserve femur and tibia as independent masks.

The editing workflow must allow:

```text
background
femur only
tibia only
femur + tibia overlap
```

Editing one structure must not automatically erase the other structure from overlapping pixels unless the user explicitly removes it.

The UI may display standard visible colours such as:

```text
Femur -> red
Tibia -> green
Overlap -> red + green
```

but the underlying annotation representation must preserve independent structure membership.

## Step 5 — Training-pool integration

The existing corrected-annotation workflow must continue to work.

When an edited annotation is approved/saved:

- preserve separate femur and tibia channels,
- preserve overlap,
- attach the corresponding source X-ray,
- add the corrected sample to the existing training pool,
- preserve existing metadata and provenance,
- avoid silently converting the annotation back to a mutually exclusive `0/1/2` mask.

If the existing training-pool format is incompatible with overlapping masks, update the minimum required storage/conversion layer while maintaining backward compatibility where possible.

## Step 6 — Fine-tuning workflow

The existing OrthoSeg/MONAI MedSAM2 fine-tuning workflow must support the updated multi-channel annotation format.

Inspect the current training implementation first.

Update it so that fine-tuning consumes the corrected annotations while preserving:

```text
Femur channel
Tibia channel
Overlap where both channels are positive
```

The fine-tuning pipeline must not discard overlap information.

Reuse the existing training entry points, job-management system, configuration files, and checkpoint-management system where possible.

Do not create a separate standalone training system unless the existing implementation genuinely cannot support the required update.

## Step 7 — Fine-tuning controls in OrthoSeg

Preserve or update the existing UI controls for model fine-tuning.

The workflow should continue to support, where already available:

- viewing the number of corrected samples in the training pool,
- starting a fine-tuning job,
- selecting training parameters already exposed by the UI,
- viewing queued/running/completed/failed status,
- viewing training logs or progress,
- identifying the output checkpoint,
- selecting or activating the resulting fine-tuned model.

Do not add unrelated training controls.

Keep the existing UX pattern rather than redesigning the model-management UI.

## Step 8 — Model lifecycle

The backend should clearly distinguish:

```text
base/reference checkpoint
current active checkpoint
newly fine-tuned checkpoints
```

Do not overwrite the existing source checkpoint automatically.

Fine-tuned models should use the existing naming/versioning convention if one already exists.

Preserve model metadata such as:

- checkpoint path,
- creation date,
- parent/base checkpoint,
- training dataset/version,
- training sample count,
- model status,
- active/inactive state.

## Step 9 — Backend inference response

Prefer explicit independent masks in the existing API representation.

Conceptually:

```json
{
  "femur_mask": "...",
  "tibia_mask": "...",
  "combined_mask": "...",
  "metadata": {
    "labels": {
      "femur": 1,
      "tibia": 2
    },
    "model": {
      "checkpoint": "...",
      "version": "..."
    }
  }
}
```

Use existing serialization conventions if they already exist.

Do not introduce a new incompatible response format unless necessary.

## Step 10 — Fine-tuning data validation

Before a sample is accepted into the training pool, validate:

```text
Femur channel values: 0 or 1
Tibia channel values: 0 or 1
```

For the combined representation:

```text
Red channel: 0 or 1
Green channel: 0 or 2
Blue channel: 0
```

Report or record:

```text
femur pixel count
tibia pixel count
overlap pixel count
```

with:

```python
overlap = femur_mask & tibia_mask
```

Reject malformed annotations rather than silently changing them.

## Step 11 — Fine-tuning validation

Add at least one end-to-end fine-tuning test:

1. Run inference on an X-ray.
2. Edit femur and/or tibia in OrthoSeg.
3. Preserve at least one overlapping region.
4. Save the corrected annotation.
5. Confirm it enters the training pool.
6. Launch a fine-tuning job through the existing workflow.
7. Confirm training starts successfully.
8. Confirm a new checkpoint is created.
9. Load the new checkpoint through the existing model-management mechanism.
10. Run inference again.
11. Confirm femur, tibia, and overlap are still represented correctly.

## Step 12 — Maintain existing architecture

Keep:

```text
MONAI backend:
/run/media/suman/Data/monai

OrthoSeg UI:
/home/suman/agentic_coding/orthoseg

Updated MedSAM2:
/run/media/suman/Data/medsam2
```

as separate components.

Do not:

- create a second backend,
- duplicate AI Fill,
- replace existing fine-tuning infrastructure,
- remove existing model-management functionality,
- retrain automatically every time an annotation is saved,
- collapse overlap into mutually exclusive classes,
- modify unrelated UI components,
- make broad architectural changes.

## Deliverables

After implementation, provide:

1. Modified file list.
2. Brief explanation of every modified file.
3. Backend run command.
4. Frontend run command.
5. Inference test command.
6. Fine-tuning test command.
7. Example inference API request/response.
8. Example fine-tuning API request/response.
9. Description of training-pool format.
10. Description of how overlapping femur/tibia annotations are stored.
11. Description of checkpoint/model lifecycle.
12. Confirmation that existing model-management and fine-tuning workflows remain operational.
13. Any backward-compatibility issues found.
14. Any migration needed for existing training-pool samples.
