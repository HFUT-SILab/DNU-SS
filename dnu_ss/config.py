"""The paper model and its seven explicitly named ablation configurations."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    ablation: str = "full"
    use_sgsb: bool = True
    use_prior: bool = True
    temporal_mode: str = "timr_gated_sert"
    num_classes: int = 300
    timesteps: int = 4
    embedding_dim: int = 64
    token_grid: int = 15
    skip_ratios: tuple = (0.0, 0.15, 0.30, 0.45)
    num_heads: int = 2
    mlp_ratio: int = 4
    stri_window: int = 3
    prior_weights: tuple = (0.4, 0.2, 0.4)
    prior_gray_window: int = 7
    prior_coherence_window: int = 7
    prior_gain_initial: float = 0.1
    prior_gain_maximum: float = 0.3
    prior_score_weight: float = 0.15
    sert_score_weight: float = 0.1
    sert_temperature: float = 0.2
    timr_decay: float = 0.6
    timr_gate_strength: float = 0.1
    teb_fusion_weight: float = 0.7
    dropout: float = 0.5
    lif_backend: str = "auto"

    def __post_init__(self):
        if self.num_classes < 1 or self.timesteps < 1:
            raise ValueError("num_classes and timesteps must be positive")
        if len(self.skip_ratios) != 4 or any(not 0 <= r < 1 for r in self.skip_ratios):
            raise ValueError("SPGTR requires four skip ratios in [0, 1)")
        if self.skip_ratios[0] != 0:
            raise ValueError("The first SPGTR block must update every position")
        if self.embedding_dim % self.num_heads or self.sert_temperature <= 0:
            raise ValueError("Invalid attention dimensions or SERT temperature")
        if not 0 < self.timr_decay < 1:
            raise ValueError("TIMR decay must be in (0, 1)")
        if (
            min(self.prior_score_weight, self.sert_score_weight) < 0
            or self.prior_score_weight + self.sert_score_weight > 1
        ):
            raise ValueError(
                "Joint scoring weights must be nonnegative and sum to at most one"
            )
        if not 0 < self.prior_gain_initial < self.prior_gain_maximum:
            raise ValueError("Prior gain initialization must be inside its bound")
        if self.lif_backend not in ("auto", "torch", "cupy"):
            raise ValueError("lif_backend must be auto, torch or cupy")
        if self.temporal_mode not in ("none", "sert", "timr_gated_sert"):
            raise ValueError("Unsupported temporal mode")
        if not self.use_prior and self.prior_score_weight != 0:
            raise ValueError("Disabled prior must have zero ranking weight")
        if self.temporal_mode == "none" and self.sert_score_weight != 0:
            raise ValueError("Disabled temporal scoring must have zero weight")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for name in ("skip_ratios", "prior_weights"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)
