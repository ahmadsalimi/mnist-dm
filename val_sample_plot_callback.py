import os
from typing import Any

import pytorch_lightning as pl
import imageio
import torch


def save_as_gif(frames: torch.Tensor, filename: str, fps: int = 100) -> None:
    # frames                                                            # T x C x H x W
    frames = frames.squeeze(1)                                          # T x H x W
    frames_min = frames.reshape(frames.shape[0], -1).min(dim=1).values[:, None, None]
    frames_max = frames.reshape(frames.shape[0], -1).max(dim=1).values[:, None, None]
    frames = (frames - frames_min) / (frames_max - frames_min)
    frames = (255 * frames).byte().cpu().numpy()
    imageio.mimsave(filename, frames, fps=fps)


class ValSamplePlot(pl.callbacks.BasePredictionWriter):

    def __init__(self, directory: str) -> None:
        super().__init__(write_interval='batch')
        self.directory = directory
        self.predictions = None

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.predictions = torch.tensor([], device=pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.predictions = torch.cat((self.predictions, outputs), dim=0)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_directory = os.path.join(self.directory, f'epoch_{trainer.current_epoch}')
        os.makedirs(epoch_directory, exist_ok=True)
        for i, pred in enumerate(self.predictions):
            filename = os.path.join(epoch_directory, f'val_sample_{i}.gif')
            save_as_gif(pred, filename)
