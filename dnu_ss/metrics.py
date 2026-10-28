"""All unordered verification pairs, ROC upper-envelope EER, and population d-prime."""

import numpy as np
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity


def pair_scores(features, labels):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels) or len(features) < 2:
        raise ValueError(
            "Features must be a nonempty [images, descriptor] array with matching labels"
        )
    count = len(labels) * (len(labels) - 1) // 2
    groups = [features[labels == label] for label in np.unique(labels)]
    positive_count = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    scores = np.empty(count, dtype=np.float64)
    genuine = np.zeros(count, dtype=np.int32)
    genuine[:positive_count] = 1
    offset = 0
    for group in groups:
        if len(group) < 2:
            continue
        similarity = cosine_similarity(group, group)
        values = similarity[np.tril_indices(len(group), -1)]
        scores[offset : offset + len(values)] = values
        offset += len(values)
    for index, group in enumerate(groups[:-1]):
        for other in groups[index + 1 :]:
            values = cosine_similarity(group, other).ravel()
            scores[offset : offset + len(values)] = values
            offset += len(values)
    if offset != count:
        raise AssertionError("Verification pair count mismatch")
    return scores, genuine


def verification_metrics(scores, pair_labels):
    scores = np.asarray(scores, dtype=np.float64)
    pair_labels = np.asarray(pair_labels, dtype=np.int32)
    if len(np.unique(pair_labels)) != 2:
        raise ValueError("Verification requires both genuine and impostor pairs")
    fpr, tpr, thresholds = metrics.roc_curve(pair_labels, scores)
    order = np.argsort(fpr, kind="mergesort")
    unique_fpr, start = np.unique(fpr[order].astype(np.float64), return_index=True)
    envelope = np.maximum.accumulate(
        np.maximum.reduceat(tpr[order].astype(np.float64), start)
    )
    eer = brentq(lambda x: 1.0 - x - np.interp(x, unique_fpr, envelope), 0.0, 1.0)
    genuine, impostor = scores[pair_labels == 1], scores[pair_labels == 0]
    pooled = np.sqrt((np.var(genuine, ddof=0) + np.var(impostor, ddof=0)) / 2.0)
    gap = float(abs(np.mean(genuine) - np.mean(impostor)))
    d_prime = (
        (9999.0 if gap else 0.0) if pooled == 0 else min(round(gap / pooled, 4), 9999.0)
    )
    result = dict(
        AUC=float(metrics.auc(fpr, tpr)) * 100,
        EER=float(eer) * 100,
        d_prime=float(d_prime),
        num_genuine=len(genuine),
        num_impostor=len(impostor),
    )
    for far, suffix in [
        (1e-6, "E6"),
        (1e-5, "E5"),
        (1e-4, "E4"),
        (1e-3, "E3"),
        (1e-2, "E2"),
        (1e-1, "E1"),
    ]:
        valid = unique_fpr <= far
        result["TAR_FAR_" + suffix] = (
            float(envelope[valid].max() if np.any(valid) else 0.0) * 100
        )
    return result, fpr, tpr, thresholds
