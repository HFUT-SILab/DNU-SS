"""Palmprint prior visualization in evaluation mode."""

from pathlib import Path
import torch


def save_prior_visualization(model, images, epoch, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.describe(images)
    finally:
        model.train(was_training)
    branch = model.sgsb
    prior = branch.last_prior[0, 0].cpu().numpy()
    selected = branch.spgtr_blocks[-1].last_selection[-1]
    mask = torch.zeros(
        selected.size(0), model.config.token_grid**2, device=selected.device
    )
    mask.scatter_(1, selected, 1.0)
    mask = mask[0].reshape(model.config.token_grid, model.config.token_grid)
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(images[0, 0].detach().cpu().numpy(), cmap="gray")
    axes[0].set_title("Input ROI")
    axes[1].imshow(prior, cmap="viridis")
    axes[1].set_title("Saliency prior")
    axes[2].imshow(mask.cpu().numpy(), cmap="viridis")
    axes[2].set_title("Last SPGTR update mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"epoch_{epoch}.png")
    plt.close(figure)
