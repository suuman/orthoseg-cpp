You are starting in the ALREADY IMPLEMENTED MONAI backend repository:

```
/run/media/suman/Data/monai
```

The existing custom annotation UI is a separate repository:

```
/home/suman/agentic_coding/orthoseg
```

The MONAI backend was already implemented for 2D X-ray Femur/Tibia segmentation.

DO NOT rebuild it.

Your task is:

1. Inspect the existing MONAI backend first and determine its ACTUAL API.
2. Inspect the existing OrthoSeg UI.
3. Integrate OrthoSeg with MONAI using the existing local HTTP API.
4. Make the FEWEST UI changes possible.
5. Preserve all existing OrthoSeg functionality.

This is an INTEGRATION task, not a redesign or refactor.

---

# 1. First inspect the existing MONAI backend

Start from:

```
/run/media/suman/Data/monai
```

Inspect the backend implementation already created.

Determine the actual:

* startup command
* host and port
* health endpoint
* segmentation endpoint
* training-case submission endpoint
* HTTP methods
* multipart field names
* segmentation response format
* mask format
* model-version response header
* error format

Verify whether endpoints approximately correspond to:

```
GET /health
GET /model/info
POST /segment
POST /training/cases
GET /training/status
```

DO NOT assume these endpoint names if the implementation differs.

USE THE ACTUAL IMPLEMENTATION.

Verify the segmentation semantics.

Expected canonical MONAI labels are:

```
0 = background
1 = femur
2 = tibia
```

Verify whether `/segment` returns:

```
image/png
```

and whether the returned mask:

* is single-channel
* represents class IDs rather than visualization colors
* has exactly the original image dimensions
* uses only 0, 1, 2

Also determine how the backend returns model version information.

Expected approximately:

```
X-Model-Version
```

but use the actual implementation.

---

# 2. Backend modification rule

Prefer making ZERO changes to:

```
/run/media/suman/Data/monai
```

The backend was already built.

If the UI integration encounters a small difference between this prompt and the real backend:

```
ADAPT THE UI CLIENT TO THE BACKEND.
```

Do NOT change the backend merely to make it conform to assumptions in this prompt.

Only make a backend compatibility change if integration is genuinely impossible without it.

If you modify anything in the MONAI backend, explicitly explain why.

---

# 3. Then inspect OrthoSeg

Inspect:

```
/home/suman/agentic_coding/orthoseg
```

Before editing anything, understand:

1. application entry point
2. language/UI framework
3. X-ray PNG loading
4. where the ORIGINAL source PNG/path/bytes are retained
5. image dimensions and bit-depth handling
6. segmentation mask representation
7. Femur label representation
8. Tibia label representation
9. existing mask import/load path
10. existing Save implementation
11. existing Export implementation
12. toolbar/actions
13. undo/redo
14. networking utilities, if any
15. config system
16. dialogs/notifications
17. build/tests
18. existing AI Fill integration
19. existing `ocv/` implementation

Pay particular attention to:

```
/home/suman/agentic_coding/orthoseg/ocv
```

This contains the existing OpenCV4+CUDA MedSAM2 implementation.

---

# 4. Strict minimal-change requirement

DO NOT:

* redesign the UI
* reorganize the repository
* rewrite segmentation storage
* create another mask/editor representation
* replace existing mask loading
* rewrite Save
* rewrite Export
* modify image rendering
* change label IDs
* change keyboard shortcuts
* change annotation behavior
* rewrite MedSAM2
* change AI Fill behavior
* copy the MONAI backend into OrthoSeg
* create another MONAI backend
* import PyTorch/MONAI directly into the UI
* perform unrelated refactoring

Prefer approximately:

```
one small MONAI client/service
    +
one AI Segment action/hook
    +
small Save/Export integration hook
```

Reuse existing code wherever possible.

---

# 5. Required architecture

Keep the repositories completely separate:

```
/home/suman/agentic_coding/orthoseg
              │
              │ localhost HTTP
              ▼
/run/media/suman/Data/monai
```

The OrthoSeg runtime should NOT depend on the filesystem path:

```
/run/media/suman/Data/monai
```

That path is for development/inspection only.

Runtime communication must occur through the local HTTP API.

---

# 6. Desired user workflow

Implement:

```
Open 2D X-ray PNG
        ↓
    AI Segment
        ↓
send ORIGINAL PNG
        ↓
existing MONAI backend
        ↓
receive Femur/Tibia mask
        ↓
validate response
        ↓
map into existing OrthoSeg
Femur/Tibia representation
        ↓
user edits normally
        ↓
optional existing AI Fill / MedSAM2
        ↓
existing Save / Export
        ↓
operation succeeds
        ↓
ask user:

"Add this corrected segmentation
 to AI training?"

      YES       NO
       ↓         ↓
    submit      finished
    original
    PNG +
    CURRENT
    corrected mask
```

Submitting the case MUST NOT start training.

Fine-tuning remains a separate backend operation/script.

---

# 7. Add/connect AI Segment

Find the appropriate existing action/toolbar system.

If an AI Segment action already exists:

```
reuse it.
```

Otherwise add ONE minimal action:

```
AI Segment
```

Do not redesign the toolbar.

Keep the distinction:

```
AI Segment
    → MONAI
    → complete Femur/Tibia segmentation

AI Fill
    → existing MedSAM2/OpenCV CUDA
    → interactive refinement
```

Both must coexist.

---

# 8. Original PNG must be sent

When AI Segment is clicked, send the ORIGINAL loaded X-ray PNG.

Do NOT send:

* viewer screenshot
* rendered canvas
* Qt/OpenGL texture
* overlay image
* segmentation visualization
* screen capture
* display-resized image
* windowed RGB screenshot

If OrthoSeg already retains:

```
original file path
```

use it if appropriate.

If it retains:

```
original PNG bytes
```

use those.

This requirement is especially important for 16-bit grayscale PNG images.

If the input is a 16-bit PNG:

```
preserve the original 16-bit PNG for MONAI.
```

Do NOT silently convert it to an 8-bit display representation merely because the UI displays it that way.

Avoid unnecessary encode/decode cycles.

---

# 9. Configure backend URL once

Use ONE configuration location.

Expected default:

```
http://127.0.0.1:8000
```

but verify the actual backend.

Use OrthoSeg's existing configuration mechanism if available.

Otherwise introduce one small configuration value.

For example:

```
MONAI_BACKEND_URL=http://127.0.0.1:8000
```

Do NOT scatter this URL throughout the code.

Do NOT hard-code:

```
/run/media/suman/Data/monai
```

into runtime UI logic.

---

# 10. Small MONAI client

Use an existing networking abstraction if one already exists.

Otherwise implement a small isolated client, conceptually:

```
MonaiClient
    health()
    segment(...)
    submitTrainingCase(...)
```

Only add:

```
modelInfo()
```

if it is actually useful/required.

Do not scatter HTTP request code among toolbar/editor components.

---

# 11. AI Segment request

Use the ACTUAL segmentation endpoint discovered in the backend.

Expected approximately:

```
POST /segment
```

multipart:

```
image=<ORIGINAL PNG>
case_id=<optional stable ID>
```

Do not invent another API.

Use reasonable inference timeout settings suitable for high-resolution X-rays.

Do not use an unrealistically short 2–5 second timeout.

Prevent accidental duplicate inference requests.

While AI Segment is running:

* disable/restrict duplicate AI Segment actions
* use OrthoSeg's existing busy/progress mechanism
* do not freeze the UI
* restore normal state after success/failure

---

# 12. Validate MONAI response before modifying annotation

Do not modify the current segmentation immediately upon receiving HTTP data.

First validate the complete result.

Verify:

```
response can be decoded
```

Verify:

```
mask width == original X-ray width
mask height == original X-ray height
```

Verify the mask is suitable as a single-channel class-ID mask.

Verify unique values are only:

```
0
1
2
```

Do not accept:

```
3
4
255
arbitrary RGB colors
```

Do NOT silently resize a mismatched mask.

Do NOT attempt to guess class identities from colors.

If validation fails:

* leave current segmentation unchanged
* show a clear error

The backend is responsible for returning original image dimensions.

---

# 13. Map MONAI labels to EXISTING UI labels

Inspect how OrthoSeg identifies Femur and Tibia.

DO NOT assume OrthoSeg uses:

```
Femur = 1
Tibia = 2
```

For example, OrthoSeg might internally use:

```
Femur = 5
Tibia = 8
```

That is completely acceptable.

Map semantically:

```
MONAI 1 → existing Femur label

MONAI 2 → existing Tibia label
```

Do not modify existing OrthoSeg label IDs.

For training export later, reverse-map:

```
OrthoSeg Femur → 1
OrthoSeg Tibia → 2
```

---

# 14. Required label handling

Before applying MONAI segmentation, verify that the current annotation setup contains:

```
Femur
Tibia
```

Do not create duplicate labels.

If OrthoSeg already has a normal mechanism for creating/ensuring required labels:

```
reuse it.
```

If safely creating the missing labels is not consistent with existing behavior, show:

```
AI Segment requires Femur and Tibia labels.
```

and do not apply the result.

Do not silently create a second Femur or Tibia class.

---

# 15. Reuse existing mask-import pathway

This is strongly preferred.

Find the current:

```
Load Mask
Import Mask
Set Segmentation
```

or equivalent implementation.

Where reasonable, feed the MONAI result through that existing pathway.

Do NOT build a separate MONAI-specific annotation layer.

The resulting segmentation must behave exactly like an ordinary OrthoSeg segmentation.

Immediately after AI Segment, users must still be able to use:

* paint
* erase
* fill
* polygon
* undo
* redo
* AI Fill
* MedSAM2
* save
* export
* any other existing editing feature

---

# 16. Protect existing annotations

If Femur/Tibia annotation already exists before AI Segment:

do not silently overwrite valuable user work.

Reuse any existing replace/import confirmation behavior.

If none exists, add only a small confirmation:

```
AI Segment will replace the current
Femur/Tibia segmentation.

Continue?

[Cancel]
[Continue]
```

Do not delete unrelated labels.

Only modify the relevant Femur/Tibia segmentation.

If OrthoSeg's undo system supports it naturally, applying a MONAI prediction should ideally be one undoable operation.

Do not redesign undo/redo to achieve this.

---

# 17. Preserve MONAI model version

If the segmentation response provides a model version, for example:

```
X-Model-Version: femur_tibia_2d_v003
```

retain that value as lightweight metadata for the current prediction/session.

Do not build a new model-management UI.

Later, when submitting the corrected case for training, include the same model version if the backend supports it.

This provides traceability:

```
model prediction
    ↓
human correction
    ↓
submitted training example
```

Handle absence of this header gracefully.

---

# 18. Existing Save/Export remains authoritative

DO NOT replace Save.

DO NOT replace Export.

Existing behavior should happen first:

```
current Save/Export
    ↓
existing operation succeeds
    ↓
optional training prompt
```

If Save/Export fails:

```
do NOT submit anything to MONAI training.
```

Any MONAI failure must not change the result of a successful Save/Export.

---

# 19. Prompt to add corrected mask to training

After an appropriate successful final Save or Export, show a small existing-style confirmation.

Suggested:

```
Segmentation saved successfully.

Use this corrected segmentation
to improve the Femur/Tibia AI model?

[Add to AI Training]
[Save Only]
```

Do NOT say:

```
Fine-tune now
Train now
Retrain MONAI
```

because selecting this option ONLY adds the case to the training pool.

---

# 20. Avoid repeated training prompts/submissions

Users may save multiple times while editing.

Do not annoy the user by repeatedly prompting or uploading the exact same unchanged annotation.

Prefer:

```
final Save
or Export
```

according to how the existing application works.

If necessary, track whether the current annotation revision has already been submitted.

If the segmentation later changes and is submitted again:

```
allow the backend to treat it as an updated revision of the same case.
```

Do not intentionally generate a new unrelated case ID for every revision.

---

# 21. Generate training mask from CURRENT corrected segmentation

This requirement is critical.

When the user chooses:

```
Add to AI Training
```

generate a NEW canonical mask from the CURRENT OrthoSeg segmentation.

DO NOT resend the original MONAI prediction.

The final mask must include all user modifications made through:

* painting
* erase
* fill
* polygon
* AI Fill
* MedSAM2
* any other existing editing mechanism

Map:

```
background → 0
existing Femur → 1
existing Tibia → 2
```

Training mask requirements:

```
uint8
single-channel
same width/height as original source image
values only {0,1,2}
```

Do not send RGB visualization colors.

Validate the generated training mask before upload.

---

# 22. Submit ORIGINAL PNG + CURRENT corrected mask

Use the ACTUAL training-case endpoint discovered in the MONAI backend.

Expected approximately:

```
POST /training/cases
```

multipart data may include:

```
image=<ORIGINAL PNG>
mask=<CURRENT corrected canonical PNG>
case_id=<stable ID>
original_filename=<original filename>
model_version=<model used for AI Segment>
```

Use the real field names expected by the backend.

Do not change backend APIs just to match this example.

---

# 23. Stable case ID

Reuse an existing OrthoSeg case/image identifier if one exists and is stable.

Otherwise derive a deterministic identifier.

A SHA-256 hash of the original PNG bytes is acceptable.

Do NOT generate a random new identifier each time the same source image is submitted.

The same original X-ray should remain the same logical training case across revisions.

---

# 24. Training submission does NOT trigger fine-tuning

The UI must NOT:

* call a training endpoint
* execute finetune.sh
* spawn PyTorch training
* start background training
* reload a newly trained model
* promote candidate checkpoints

It only submits approved corrected data.

Training remains separate in:

```
/run/media/suman/Data/monai
```

---

# 25. User feedback

On successful submission:

```
Corrected segmentation added to AI training data.
```

If the backend returns something like:

```
new_cases_since_last_training
```

it is acceptable to display:

```
New cases awaiting training: 14
```

Do NOT display:

```
Model updated
```

because the model has not yet been trained.

---

# 26. Failed training upload must not affect saved work

If:

```
Save/Export succeeds
```

but:

```
POST training case fails
```

do NOT undo the saved/exported result.

Show a small message such as:

```
Segmentation was saved successfully,
but it could not be added to AI training.
```

Allow retry where consistent with the existing UI.

---

# 27. Backend unavailable

MONAI is an optional local service from OrthoSeg's perspective.

If MONAI is not running:

```
AI Segment should fail gracefully.
```

Show approximately:

```
MONAI AI Segment service is unavailable.
Start the local MONAI backend and try again.
```

Do not crash.

The following must continue working normally:

* manual annotation
* AI Fill / MedSAM2
* mask load
* Save
* Export

---

# 28. Preserve MedSAM2/AI Fill completely

Do not change:

```
/home/suman/agentic_coding/orthoseg/ocv
```

unless an unrelated compile issue absolutely requires a trivial fix.

Expected combined workflow:

```
MONAI AI Segment
        ↓
initial full segmentation
        ↓
existing OrthoSeg editor
        ↓
optional existing MedSAM2 AI Fill
        ↓
manual refinement
        ↓
Save / Export
        ↓
optionally send corrected result
to MONAI training pool
```

This is intentional.

---

# 29. Error handling

Handle at minimum:

* backend offline
* connection refused
* request timeout
* MONAI HTTP error
* malformed response
* invalid PNG response
* mask dimension mismatch
* unsupported mask values
* missing Femur label
* missing Tibia label
* training upload failure

None should crash OrthoSeg.

On inference failure:

```
preserve current segmentation.
```

On training-upload failure:

```
preserve successful Save/Export.
```

---

# 30. Focused tests only

Do not create a huge new testing framework.

Reuse existing testing conventions.

At minimum test:

### Client

* backend health success
* backend unavailable
* inference success
* timeout/error handling
* mask decoding
* model-version header if supported

### Mask validation

* valid 0/1/2 accepted
* value outside 0/1/2 rejected
* wrong dimensions rejected
* invalid response rejected

### Label mapping

Test even if internal UI IDs are not 1/2.

Verify:

```
MONAI 1 → Femur
MONAI 2 → Tibia
```

and reverse:

```
Femur → canonical 1
Tibia → canonical 2
```

### Training submission

Verify:

* original source PNG is submitted
* CURRENT corrected segmentation is submitted
* initial MONAI prediction is NOT accidentally submitted after edits
* Save Only submits nothing
* failed upload does not undo successful Save/Export

### Regression

Run enough existing tests/build checks to ensure:

* painting works
* erase works
* mask loading works
* AI Fill works
* Save works
* Export works

---

# 31. Actual integration validation

After implementation:

1. Start MONAI using the REAL startup method found in:

   ```
   /run/media/suman/Data/monai
   ```

2. Start OrthoSeg normally.

3. Test:

   ```
   load X-ray PNG
        ↓
   AI Segment
        ↓
   MONAI returns prediction
        ↓
   Femur/Tibia appear correctly
        ↓
   edit segmentation
        ↓
   optionally use existing AI Fill
        ↓
   Save/Export
        ↓
   Add to AI Training
        ↓
   backend receives:
       ORIGINAL X-ray PNG
       CURRENT corrected mask
   ```

4. Verify submitted mask contains user edits.

5. Verify training was NOT started.

6. Verify MedSAM2/AI Fill still behaves exactly as before.

---

# 32. Final acceptance criteria

The implementation is complete when:

1. OrthoSeg runs as before.
2. Existing X-ray loading is unchanged.
3. Existing annotation tools work.
4. Existing AI Fill/MedSAM2 works.
5. AI Segment calls the already-built MONAI service.
6. Original source PNG is sent.
7. 16-bit source PNG is not accidentally reduced to the UI display representation.
8. MONAI mask is validated before application.
9. Dimensions match exactly.
10. Only canonical classes 0/1/2 are accepted.
11. MONAI Femur maps to existing Femur.
12. MONAI Tibia maps to existing Tibia.
13. Existing label IDs are unchanged.
14. Prediction is editable using existing tools.
15. Save remains unchanged.
16. Export remains unchanged.
17. Add to AI Training sends original PNG.
18. Add to AI Training sends the CURRENT corrected mask.
19. Training mask is single-channel uint8 0/1/2.
20. Same case can be revised without creating unnecessary duplicates.
21. UI does not trigger fine-tuning.
22. MONAI failures do not affect normal annotation.
23. `/run/media/suman/Data/monai` has not been copied into OrthoSeg.
24. No second MONAI backend has been created.
25. Changes to OrthoSeg are minimal.

---

# 33. Final report

At completion provide a concise report containing:

1. Actual MONAI endpoints discovered.
2. Actual MONAI startup command.
3. OrthoSeg files modified.
4. MONAI files modified — preferably NONE.
5. Where the MONAI base URL is configured.
6. How AI Segment sends the original image.
7. How response masks are validated.
8. How MONAI Femur/Tibia classes map to OrthoSeg.
9. How current corrected segmentation is converted back to canonical 0/1/2.
10. How training submission works.
11. How duplicate/revised cases are handled.
12. Confirmation that UI does NOT start fine-tuning.
13. Confirmation that `ocv/` / MedSAM2 remains unchanged.
14. Build/test results.
15. Exact commands to run MONAI and OrthoSeg.

IMPORTANT:

MAKE THE FEWEST CHANGES POSSIBLE.

Do not stop after analysis.

Implement and test the integration.

Do not refactor unrelated functionality.

Do not leave core integration functionality as TODO.

