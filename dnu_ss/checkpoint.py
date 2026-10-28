"""Strict loading for the new, paper-named checkpoint format."""

from pathlib import Path
import torch
from .config import ModelConfig
from .models import DNUSS


def load_checkpoint(path, device="cpu", lif_backend=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "dnu_ss_v1":
        raise ValueError("Expected a DNU-SS checkpoint with format dnu_ss_v1")
    values = dict(payload["model_config"])
    if lif_backend is not None:
        values["lif_backend"] = lif_backend
    elif str(device) == "cpu":
        values["lif_backend"] = "torch"
    model = DNUSS(ModelConfig.from_dict(values))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def save_checkpoint(path, model, training):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        dict(
            format="dnu_ss_v1",
            model_config=model.config.to_dict(),
            state_dict=model.state_dict(),
            training=training,
        ),
        temporary,
    )
    temporary.replace(target)
