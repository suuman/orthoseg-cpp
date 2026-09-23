import numpy as np


def segmentation_metrics(predictions):
    """Dataset-global foreground Dice/IoU. Both-empty classes are undefined, not perfect."""
    intersection = np.zeros(2, dtype=np.float64)
    predicted = np.zeros(2, dtype=np.float64)
    actual = np.zeros(2, dtype=np.float64)
    for pred, target in predictions:
        for i, label in enumerate((1, 2)):
            p, t = pred == label, target == label
            intersection[i] += (p & t).sum()
            predicted[i] += p.sum()
            actual[i] += t.sum()
    result = {}
    scores = []
    for i, name in enumerate(("femur", "tibia")):
        denominator = predicted[i] + actual[i]
        dice = float(2 * intersection[i] / denominator) if denominator else None
        result[f"{name}_dice"] = dice
        union = denominator - intersection[i]
        result[f"{name}_iou"] = float(intersection[i] / union) if union else None
        if dice is not None:
            scores.append(dice)
    result["mean_foreground_dice"] = float(np.mean(scores)) if scores else None
    return result
