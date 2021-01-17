from typing import List

import torch
from torch import nn, optim
from matplotlib import pyplot as plt
import seaborn as ses
import pytorch_lightning as pl

ses.set(rc={"font.size": 16})


class ClassificationModelTrainer(pl.LightningModule):

    def __init__(self, model, optimizer_params: dict, sheduler_params: dict, class_labels: List[str]):
        assert len(class_labels) > 1
        super().__init__()
        self.cls_model = model
        self._optimizer_params = optimizer_params
        self._sheduler_params = sheduler_params
        self._train_accuracy = pl.metrics.Accuracy(compute_on_step=True)
        self._test_conf_matrix = pl.metrics.ConfusionMatrix(len(class_labels), compute_on_step=False)
        self._class_labels = class_labels

    def forward(self, batch):
        return self.cls_model.forward(batch)

    def _compute_loss(self, prediction, true):
        return nn.functional.nll_loss(prediction, true, reduction="sum")

    def training_step(self, batch, batch_idx):
        prediction = self.cls_model(batch)
        loss = self._compute_loss(prediction, batch.y)

        true_in_batch = batch.y
        pred_in_batch = prediction.argmax(dim=1)
        self._train_accuracy(pred_in_batch, true_in_batch)
        self.log("Train/loss_cross_entropy", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("Train/overall_accuracy", self._train_accuracy.compute(), logger=True)

    def validation_step(self, batch, batch_idx):
        pred = self.cls_model(batch)
        true = batch.y
        val_loss = self._compute_loss(pred, true)
        pred = pred.argmax(dim=1)
        self._test_conf_matrix(pred, true)
        self.log("Test/loss_cross_entropy", val_loss, on_step=False, on_epoch=True, logger=True)

    def validation_epoch_end(self, outputs):
        conf_matrix = self._test_conf_matrix.compute().cpu()
        total = conf_matrix.sum().item()

        fig = plt.figure(1, figsize=(10, 10))
        ax = fig.add_subplot(111)
        ses.heatmap(conf_matrix.type(torch.int32), annot=True, cmap="coolwarm_r", fmt="d", square=True,
                    yticklabels=self._class_labels,
                    xticklabels=self._class_labels, ax=ax)
        logger = self.logger.experiment
        logger.add_figure("Test/confusion_matrix", fig, global_step=self.current_epoch)

        accuracy = torch.trace(conf_matrix) / total
        self.log("Test/overall_accuracy", accuracy)
        total_samples_per_class = conf_matrix.sum(axis=1)
        acc_per_class = {label: conf_matrix[index][index] / total_samples_per_class[index]
                         for index, label in enumerate(self._class_labels)}
        logger.add_scalars("Test/accuracy_per_class", acc_per_class, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self._optimizer_params.__dict__)
        sheduler = optim.lr_scheduler.StepLR(optimizer, **self._sheduler_params.__dict__)
        return [optimizer], [sheduler]
