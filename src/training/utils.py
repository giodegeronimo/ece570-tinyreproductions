"""Training and evaluation helpers for experiment notebooks/scripts."""

import torch
from tqdm.auto import tqdm


def train_loop(model, dataloader, optimizer, loss_fn, device) -> float:
    """
    Run one training epoch.

    Args:
        model: nn.Module under training.
        dataloader: iterable of (masked, full) image tensors.
        optimizer: optimizer instance.
        loss_fn: callable producing a scalar loss.
        device: torch.device to send tensors/model to.
    """

    model.train()
    total_loss = 0.0
    for images_masked, images_full in tqdm(dataloader, desc="train", leave=False):
        images_masked = images_masked.to(device)
        images_full = images_full.to(device)

        optimizer.zero_grad()
        outputs = model(images_masked)
        loss = loss_fn(outputs, images_full)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def test_loop(model, dataloader, loss_fn, device) -> float:
    """Evaluate the model on validation data."""

    model.eval()
    total_loss = 0.0
    for images_masked, images_full in tqdm(dataloader, desc="val", leave=False):
        images_masked = images_masked.to(device)
        images_full = images_full.to(device)
        outputs = model(images_masked)
        loss = loss_fn(outputs, images_full)
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)
