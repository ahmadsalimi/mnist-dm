from functools import cached_property
from typing import Union, Tuple, List

import torch
from pytorch_lightning import LightningModule

from unet.model import UNet


class MNISTDiffusionUNetModule(LightningModule):
    def __init__(self, T: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 1e-2,
                 n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2),
                 n_blocks: int = 2,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6):
        super().__init__()
        self.epsilon_theta = UNet(n_channels=n_channels, ch_mults=ch_mults, n_blocks=n_blocks)
        self.hparams.update(T=T,
                            beta_start=beta_start,
                            beta_end=beta_end,
                            n_channels=n_channels,
                            ch_mults=ch_mults,
                            n_blocks=n_blocks,
                            lr=lr,
                            weight_decay=weight_decay)

    @cached_property
    def betas(self) -> torch.Tensor:
        return torch.linspace(self.hparams['beta_start'],
                              self.hparams['beta_end'],
                              self.hparams['T'],
                              device=self.device)

    @cached_property
    def alphas(self) -> torch.Tensor:
        return 1 - self.betas

    @cached_property
    def alphas_bar(self) -> torch.Tensor:
        return self.alphas.cumprod(dim=0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.epsilon_theta(x, t)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_0, _ = batch
        x_0 = x_0.to(self.device)
        t = torch.randint(low=0, high=self.hparams['T'], size=(x_0.shape[0],), device=self.device)
        epsilon = torch.randn_like(x_0)
        alpha_bar_t = self.alphas_bar[t][:, None, None, None]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        epsilon_theta = self.epsilon_theta(x_t, t)
        loss = (epsilon - epsilon_theta).norm(dim=0).pow(2).mean()
        self.log('train_loss', loss)
        return loss

    def sample(self, x_t: torch.Tensor) -> torch.Tensor:
        x = [x_t]
        for t in torch.arange(self.hparams['T'] - 1, -1, -1, device=self.device):
            z = torch.randn_like(x_t) if t else 0
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_bar[t]
            t = t.repeat(x_t.shape[0])
            epsilon_theta = self.epsilon_theta(x_t, t)
            x_t = 1 / torch.sqrt(alpha_t) * (
                x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta
            ) + torch.sqrt(1 - alpha_t) * z
            x.append(x_t)
        return torch.stack(x, dim=1)    # B x T x C x H x W

    def validation_step(self, x_t: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_0 = self.sample(x_t)
        return x_0

    def configure_optimizers(self):
        return torch.optim.Adam(self.epsilon_theta.parameters(),
                                lr=self.hparams['lr'],
                                weight_decay=self.hparams['weight_decay'])
