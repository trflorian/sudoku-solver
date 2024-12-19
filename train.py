import torch

from torchvision.datasets import MNIST
from torchvision.transforms import v2 as T

import lightning as L

from model import DigitClassifier

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download data
        MNIST(root="data", train=True, download=True)
        MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            transform = T.Compose(
                [
                    T.ToTensor(),  # convert to 3 channels
                    T.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            )
            mnist_full = MNIST(
                root="data",
                train=True,
                transform=transform,
            )
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                mnist_full, [55000, 5000]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size)



data = MNISTDataModule(batch_size=1024)

trainer = L.Trainer(max_epochs=3, max_steps=100)

model = DigitClassifier()

trainer.fit(model, data)

# print final model metrics
print(trainer.logged_metrics)

# save the model
trainer.save_checkpoint("model.ckpt")