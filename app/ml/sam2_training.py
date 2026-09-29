"""MedSAM2 adapter for the existing trainer, job runner and candidate lifecycle."""
import numpy as np
import torch
import torch.nn.functional as F

from app.ml.metrics import segmentation_metrics
from app.ml.preprocessing import class_mask, decode_png
from app.ml.sam2_predictor import Sam2Predictor, square_input, prompt_canvas, mask_box, mask_logits


class Sam2TrainingAdapter:
    def __init__(self, model, config, device):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.model, self.config, self.device = model, config, device
        # Freeze the large image/memory encoders; train prompt encoder and mask decoder.
        model.requires_grad_(False)
        model.sam_prompt_encoder.requires_grad_(True)
        model.sam_mask_decoder.requires_grad_(True)
        self.predictor = SAM2ImagePredictor(model)

    def train_epoch(self, pairs, optimizer, scaler, amp, epoch):
        self.model.eval()
        self.model.sam_prompt_encoder.train()
        self.model.sam_mask_decoder.train()
        losses = []
        order = np.random.default_rng(self.config['training']['seed'] + epoch).permutation(len(pairs))
        batch_size = self.config['training']['batch_size']
        optimizer.zero_grad(set_to_none=True)
        for n, index in enumerate(order):
            image, mask = pairs[index]
            raw = decode_png(image.read_bytes(), self.config['limits'])
            target = decode_png(mask.read_bytes(), self.config['limits'], mask=True)
            rgb, geometry = square_input(raw, self.config)
            with torch.autocast(device_type=self.device.type, enabled=amp):
                self.predictor.set_image(rgb)  # Encoder features are intentionally frozen.
                per_bone = []
                for label in (1, 2):
                    binary = prompt_canvas(class_mask(target, label), geometry, rgb.shape[0])
                    box = mask_box(binary)
                    # Include negative examples rather than silently ignoring empty structures.
                    if box is None:
                        box = np.array([0, 0, rgb.shape[1]-1, rgb.shape[0]-1], np.float32)
                    coords = torch.as_tensor(box, device=self.device)[None]
                    coords = self.predictor._transforms.transform_boxes(coords, True, rgb.shape[:2]).reshape(1, 2, 2)
                    points = (coords, torch.tensor([[2, 3]], dtype=torch.int, device=self.device))
                    mask_input = None
                    # Alternate coarse-mask and box prompts; both keep the other bone independent.
                    if (epoch + n + label) % 2 == 0:
                        mask_input = torch.from_numpy(mask_logits(binary)[None]).to(self.device)
                    sparse, dense = self.model.sam_prompt_encoder(points=points, boxes=None, masks=mask_input)
                    low, quality, _, _ = self.model.sam_mask_decoder(
                        image_embeddings=self.predictor._features['image_embed'],
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                        multimask_output=False, repeat_image=False,
                        high_res_features=self.predictor._features['high_res_feats'])
                    truth = torch.from_numpy(binary.copy()).to(self.device, dtype=torch.float32)[None, None]
                    truth = F.interpolate(truth, size=low.shape[-2:], mode='nearest')
                    probability = low.float().sigmoid()
                    dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
                    predicted = low.detach() > 0
                    foreground = truth > 0
                    iou = ((predicted & foreground).sum().float() + 1) / ((predicted | foreground).sum().float() + 1)
                    per_bone.append(F.binary_cross_entropy_with_logits(low.float(), truth) + dice +
                                    0.1 * F.mse_loss(quality.float(), iou.expand_as(quality)))
                loss = torch.stack(per_bone).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite MedSAM2 training loss')
            group_size = min(batch_size, len(order) - (n // batch_size) * batch_size)
            scaler.scale(loss / group_size).backward()
            if (n + 1) % batch_size == 0 or n + 1 == len(order):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        return losses

    def evaluate(self, pairs):
        self.model.eval()
        engine = Sam2Predictor(self.config)
        engine._predictor, engine._version = self.predictor, 'validation'
        def predictions():
            for image, label in pairs:
                raw = decode_png(image.read_bytes(), self.config['limits'])
                truth = decode_png(label.read_bytes(), self.config['limits'], mask=True)
                boxes = {}
                for bone in (1, 2):
                    box = mask_box(class_mask(truth, bone))
                    if box is not None:
                        box = box.astype(int)
                        # The endpoint uses positive-area inclusive boxes.
                        if box[0] == box[2]:
                            box[0], box[2] = max(0, box[0]-1), min(raw.shape[1]-1, box[2]+1)
                        if box[1] == box[3]:
                            box[1], box[3] = max(0, box[1]-1), min(raw.shape[0]-1, box[3]+1)
                        boxes[bone] = box.tolist()
                pred = engine.predict(raw, boxes)[0] if boxes else np.zeros((*raw.shape, 3), np.uint8)
                yield pred, truth
        return segmentation_metrics(predictions())
