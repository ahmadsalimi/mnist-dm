import torch
from torch.utils.data import Dataset


class NoiseDataset(Dataset):

    def __init__(self, *shape: int, num_samples: int = 20) -> None:
        super().__init__()
        self.shape = shape
        self.num_samples = num_samples

    def __getitem__(self, _: int) -> torch.Tensor:
        return torch.randn(*self.shape)

    def __len__(self) -> int:
        return self.num_samples
