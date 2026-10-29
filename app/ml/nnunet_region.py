"""Load the two-head region model without requiring its custom training class."""
import json
from pathlib import Path

import numpy as np
import torch

REGION_LABELS = {"background": 0, "femur": [1, 3], "tibia": [2, 3]}


def load_region_network(root, checkpoint="checkpoint_best.pth", fold=0):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    root = Path(root)
    dataset = json.loads((root / "dataset.json").read_text())
    if dataset.get("labels") != REGION_LABELS or dataset.get("regions_class_order") != [1, 2]:
        raise ValueError("Expected independent femur [1,3] and tibia [2,3] regions, in that order")
    plans = PlansManager(str(root / "plans.json"))
    configuration = plans.get_configuration("2d")
    arch = configuration.configuration["architecture"]
    if arch["network_class_name"] != "dynamic_network_architectures.architectures.unet.PlainConvUNet":
        raise ValueError("Unexpected region model architecture")
    if (list(configuration.patch_size) not in ([2048, 768], [2048, 256]) or
            configuration.configuration["normalization_schemes"] != ["ZScoreNormalization"] or
            configuration.configuration["use_mask_for_norm"] != [False]):
        raise ValueError("Region model does not match the OpenCV 2048-high, 256/768-wide normalization contract")
    network = get_network_from_plans(arch["network_class_name"], arch["arch_kwargs"],
                                    arch["_kw_requires_import"], 1, 2,
                                    allow_init=False, deep_supervision=False)
    # Checkpoints are locally supplied, trusted training artifacts.
    weights = torch.load(root / f"fold_{fold}" / checkpoint, map_location="cpu", weights_only=False)
    network.load_state_dict(weights["network_weights"], strict=True)
    return network.eval(), plans, configuration, dataset


def region_probabilities_to_rgb(probabilities):
    """Canonical discrete RGB: R=1 femur, G=2 tibia; overlap is (1,2,0)."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 4 and probabilities.shape[1] == 1:
        probabilities = probabilities[:, 0]
    if probabilities.ndim != 3 or probabilities.shape[0] != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Expected two finite region probability maps")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid region probabilities")
    result = np.zeros((*probabilities.shape[1:], 3), np.uint8)
    result[..., 0] = probabilities[0] > 0.5
    result[..., 1] = (probabilities[1] > 0.5).astype(np.uint8) * 2
    return result
