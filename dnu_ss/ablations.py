"""Named paper ablations, shared by training, evaluation and result validation."""

from dataclasses import replace
from .config import ModelConfig

PRESETS = {
    "teb_only": dict(
        use_sgsb=False,
        use_prior=False,
        temporal_mode="none",
        skip_ratios=(0.0, 0.0, 0.0, 0.0),
        prior_score_weight=0.0,
        sert_score_weight=0.0,
    ),
    "dense_no_prior": dict(
        use_sgsb=True,
        use_prior=False,
        temporal_mode="none",
        skip_ratios=(0.0, 0.0, 0.0, 0.0),
        prior_score_weight=0.0,
        sert_score_weight=0.0,
    ),
    "prune_stri": dict(
        use_sgsb=True,
        use_prior=False,
        temporal_mode="none",
        skip_ratios=(0.0, 0.15, 0.30, 0.45),
        prior_score_weight=0.0,
        sert_score_weight=0.0,
    ),
    "dense_prior": dict(
        use_sgsb=True,
        use_prior=True,
        temporal_mode="none",
        skip_ratios=(0.0, 0.0, 0.0, 0.0),
        prior_score_weight=0.15,
        sert_score_weight=0.0,
    ),
    "prune_stri_prior": dict(
        use_sgsb=True,
        use_prior=True,
        temporal_mode="none",
        skip_ratios=(0.0, 0.15, 0.30, 0.45),
        prior_score_weight=0.15,
        sert_score_weight=0.0,
    ),
    "prune_stri_prior_sert": dict(
        use_sgsb=True,
        use_prior=True,
        temporal_mode="sert",
        skip_ratios=(0.0, 0.15, 0.30, 0.45),
        prior_score_weight=0.15,
        sert_score_weight=0.1,
    ),
    "full": dict(
        use_sgsb=True,
        use_prior=True,
        temporal_mode="timr_gated_sert",
        skip_ratios=(0.0, 0.15, 0.30, 0.45),
        prior_score_weight=0.15,
        sert_score_weight=0.1,
    ),
}


def make_config(ablation="full", **values):
    if ablation not in PRESETS:
        raise ValueError(f"Unknown paper ablation: {ablation}")
    return replace(ModelConfig(**values), ablation=ablation, **PRESETS[ablation])


def validate_config(config):
    if config.ablation not in PRESETS:
        raise ValueError(f"Unknown paper ablation: {config.ablation}")
    for key, expected in PRESETS[config.ablation].items():
        if getattr(config, key) != expected:
            raise ValueError(f"Ablation {config.ablation}: {key} must be {expected!r}")
    if not config.use_sgsb:
        return []
    return [max(1, int((1 - r) * config.token_grid**2)) for r in config.skip_ratios]
