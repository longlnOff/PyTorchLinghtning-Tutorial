import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, lr):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=10, average="macro"
        )
        self.lr = lr

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    # We can use _common_step() to reduce code duplication
    def _common_step(self, batch, batch_idx):
        x, y = batch
        # Flatten image
        x = x.reshape(x.shape[0], -1)
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, y)  # Taking more computation time
        f1_score = self.f1_score(logits, y)  # Taking more computation time
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        # Flatten image
        x = x.reshape(x.shape[0], -1)
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
