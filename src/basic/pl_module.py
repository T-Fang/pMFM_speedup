import torch
from torch import optim

import pytorch_lightning as pl


class PlModule(pl.LightningModule):
    """
    Our custom pytorch lightning module with pre-defined methods and attributes such as metrics used during evaluation.
    Sub-class of this module should still implement forward() method at bare minimum
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def _loss2metrics(self, loss):
        individual_mse = torch.mean(loss, dim=0)

        FC_CORR_mse = individual_mse[0]
        FC_L1_mse = individual_mse[1]
        FCD_KS_mse = individual_mse[2]

        mse_loss = torch.mean(individual_mse)

        return mse_loss, FC_CORR_mse, FC_L1_mse, FCD_KS_mse

    def log_epoch_metric(self, y_hat: torch.Tensor, y: torch.Tensor, phase: str = 'train'):
        loss = self.criterion(y_hat, y)
        mse_loss, FC_CORR_mse, FC_L1_mse, FCD_KS_mse = self._loss2metrics(loss)
        self.log(f"{phase}_epoch/mse_loss", mse_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{phase}_epoch/FC_CORR_mse", FC_CORR_mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{phase}_epoch/FC_L1_mse", FC_L1_mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{phase}_epoch/FCD_KS_mse", FCD_KS_mse, prog_bar=False, on_step=False, on_epoch=True)

    def get_y_hat_and_y(self, batch):
        x, y = batch
        y_hat = self(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_y_hat_and_y(batch)

        loss = self.loss_fn(y_hat, y)
        self.log("train_step/mse_loss", loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.get_y_hat_and_y(batch)
        self.log_epoch_metric(y_hat, y, phase='val')

    def test_step(self, batch, batch_idx):
        y_hat, y = self.get_y_hat_and_y(batch)
        self.log_epoch_metric(y_hat, y, phase='test')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [scheduler]
