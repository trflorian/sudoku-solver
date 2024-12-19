import torch

from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC

import timm

import lightning as L

class DigitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "mobilenetv3_small_100", pretrained=True, num_classes=10, global_pool="avg"
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=10),
                "precision": Precision(task="multiclass", num_classes=10),
                "recall": Recall(task="multiclass", num_classes=10),
                "f1": F1Score(task="multiclass", num_classes=10),
                "auroc": AUROC(task="multiclass", num_classes=10),
            }
        )

        self.train_metrics = self.metrics.clone(prefix="train/")
        self.val_metrics = self.metrics.clone(prefix="val/")

    def forward(self, x):
        logits = self.model(x)
        return torch.nn.functional.log_softmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Loss
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Metrics
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Loss
        loss = self.loss(y_hat, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Metrics
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-2)