import numpy as np
import cv2

import torch

from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import v2 as T
from torchvision.transforms import RandAugment

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import albumentations as A

from model import DigitClassifier


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        data_dir: str = "data",
        additional_data_dir: str = None,
        transform_preprocess: T.Compose | None = None,
        augment_train: A.Compose | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.transform_train = T.Compose(
            [
                T.Lambda(lambda x: augment_train(image=np.array(x))["image"]),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                transform_preprocess,
            ]
        )
        self.transform_val = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                transform_preprocess,
            ]
        )

        self.additional_data_dir = additional_data_dir

    def prepare_data(self):
        # download data
        MNIST(root="data", train=True, download=True)
        MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Load full training dataset with no transform (to split indices)
            full_train_mnist = MNIST(root=self.data_dir, train=True, download=False)

            full_train_additional = None
            if self.additional_data_dir is not None:
                full_train_additional = ImageFolder(
                    root=self.additional_data_dir,
                )

            full_train = torch.utils.data.ConcatDataset(
                [full_train_mnist, full_train_additional]
                if full_train_additional
                else [full_train_mnist]
            )

            val_size = 1024
            train_size = len(full_train) - val_size

            # split indices randomly
            train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(
                full_train, [train_size, val_size]
            )

            # create datasets
            self.train_ds = MapDataset(
                train_dataset_raw, lambda x: (self.transform_train(x[0]), x[1])
            )
            self.val_ds = MapDataset(
                val_dataset_raw, lambda x: (self.transform_val(x[0]), x[1])
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=4, shuffle=False
        )


data = MNISTDataModule(
    batch_size=1024,
    transform_preprocess=T.Compose(
        [
            T.Resize((28, 28)),  # resize to 28x28
            T.Normalize((0.1307,), (0.3081,)),  # mean and std of MNIST dataset
            T.Lambda(
                lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
            ),  # add channel dimension
        ]
    ),
    augment_train=A.Compose(
        [
            A.InvertImg(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5),
            A.PixelDropout(dropout_prob=0.1, drop_value=0),
            A.PixelDropout(dropout_prob=0.1, drop_value=255),
        ]
    ),
    additional_data_dir="data/digits",
)


trainer = L.Trainer(
    max_epochs=10,
    callbacks=[
        EarlyStopping(monitor="val/loss", patience=3),
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="./",
            filename="model",
            save_top_k=1,
            enable_version_counter=False,
        ),
    ],
)

model = DigitClassifier()

# visualize sample data
data.setup()
sample = next(iter(data.train_dataloader()))
print(sample[0].shape, sample[1].shape)

sample_digits = [sample[0][sample[1] == i] for i in range(10)]
sample_digits = [sample_digits[i][:10] for i in range(10)]

# convert to numpy and display
sample_digits = [x.permute(0, 2, 3, 1).numpy() * 255 for x in sample_digits]

sample_digits_img = np.vstack([np.hstack(row) for row in sample_digits])


cv2.imwrite("sample_digits.png", sample_digits_img)
cv2.imshow("Sample Digits", sample_digits_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

trainer.fit(model, data)

# print final model metrics
print(trainer.logged_metrics)
