from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from noise_dataset import NoiseDataset


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = './.data-cache/mnist',
                 batch_size: int = 32,
                 num_workers: int = 0,
                 val_num_samples: int = 20):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_samples = val_num_samples
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage: str) -> None:
        self.mnist_train = MNIST(self.data_dir,
                                 train=True,
                                 download=True,
                                 transform=self.transform)
        self.noise_gen = NoiseDataset(1, 28, 28, num_samples=self.val_num_samples)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        # val data is just noise
        return DataLoader(self.noise_gen,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
