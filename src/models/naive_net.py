from typing import List

import torch
import torch.nn as nn

import optuna

from pytorch_lightning import Trainer, loggers as pl_loggers

from src.utils.training_utils import get_trainer_callbacks
from src.basic.constants import LOG_INTERVAL
from src.basic.pl_module import PlModule
from src.basic.cost_type import CostType
from src.models.extract_SC_feat import ExtractScFeatMLP


class NaiveNet(PlModule):

    def __init__(self, output_dims: List[int], extract_SC_feat_net=None, param_vector_len=205):
        super().__init__()
        self.output_dims = output_dims
        self.use_SC_feat = extract_SC_feat_net is not None
        self.extract_SC_feat_net = extract_SC_feat_net

        input_dim = extract_SC_feat_net.SC_feat_dim + param_vector_len if self.use_SC_feat else param_vector_len

        layers = []
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, CostType.num_of_cost_type()))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            batch_SCs, batch_params = x
        else:
            batch_SCs, batch_params = None, x

        if self.use_SC_feat:
            SC_feats = self.extract_SC_feat_net(batch_SCs)
            combined_feat = torch.cat((SC_feats, batch_params), dim=1)
        else:
            combined_feat = batch_params

        y_hat = self.layers(combined_feat)
        return y_hat


def objective_naive_net_no_SC_use_coef(trial: optuna.Trial, training_loader, validation_loader, save_dir: str) -> float:
    # training params
    max_epochs = 100
    save_top_k = 5

    # We optimize the number of layers and hidden units in each layer of naive net.
    num_of_naive_net_layers = trial.suggest_int("num_of_naive_net_layers", 1, 10)
    naive_net_output_dims = [
        trial.suggest_int(f"naive_net_output_dim_l{i}", 4, 256, log=True) for i in range(num_of_naive_net_layers)
    ]

    model = NaiveNet(naive_net_output_dims, param_vector_len=10)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, save_top_k),
        #   limit_train_batches=4,
        #   limit_val_batches=4,
        #   limit_test_batches=4,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=LOG_INTERVAL,
        enable_progress_bar=False)

    hyperparams = dict(num_of_naive_net_layers=num_of_naive_net_layers, naive_net_output_dims=naive_net_output_dims)
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, training_loader, validation_loader)

    return trainer.callback_metrics["val_epoch/mse_loss"].item()


def objective_naive_net_no_SC(trial: optuna.Trial, training_loader, validation_loader, save_dir: str) -> float:
    # training params
    max_epochs = 100
    save_top_k = 5

    # We optimize the number of layers and hidden units in each layer of naive net.
    num_of_naive_net_layers = trial.suggest_int("num_of_naive_net_layers", 1, 10)
    naive_net_output_dims = [
        trial.suggest_int(f"naive_net_output_dim_l{i}", 4, 256, log=True) for i in range(num_of_naive_net_layers)
    ]

    model = NaiveNet(naive_net_output_dims)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, save_top_k),
        #   limit_train_batches=4,
        #   limit_val_batches=4,
        #   limit_test_batches=4,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=LOG_INTERVAL,
        enable_progress_bar=False)

    hyperparams = dict(num_of_naive_net_layers=num_of_naive_net_layers, naive_net_output_dims=naive_net_output_dims)
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, training_loader, validation_loader)

    return trainer.callback_metrics["val_epoch/mse_loss"].item()


def objective_naive_net_with_SC_use_coef(trial: optuna.Trial, training_loader, validation_loader,
                                         save_dir: str) -> float:
    # training params
    max_epochs = 100
    save_top_k = 5

    # We optimize the number of layers and hidden units in each layer of naive net as well as that of SC feature extraction net.
    num_of_extract_SC_feat_layers = trial.suggest_int("num_of_extract_SC_feat_layers", 1, 5)
    extract_SC_feat_output_dims = [
        trial.suggest_int(f"extract_SC_feat_output_dim_l{i}", 4, 256, log=True)
        for i in range(num_of_extract_SC_feat_layers)
    ]
    num_of_naive_net_layers = trial.suggest_int("num_of_naive_net_layers", 1, 5)
    naive_net_output_dims = [
        trial.suggest_int(f"naive_net_output_dim_l{i}", 4, 256, log=True) for i in range(num_of_naive_net_layers)
    ]

    extract_SC_feat_net = ExtractScFeatMLP(extract_SC_feat_output_dims)
    model = NaiveNet(naive_net_output_dims, extract_SC_feat_net=extract_SC_feat_net, param_vector_len=10)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, save_top_k),
        #   limit_train_batches=4,
        #   limit_val_batches=4,
        #   limit_test_batches=4,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=LOG_INTERVAL,
        enable_progress_bar=False)

    hyperparams = dict(num_of_extract_SC_feat_layers=num_of_extract_SC_feat_layers,
                       extract_SC_feat_output_dims=extract_SC_feat_output_dims,
                       num_of_naive_net_layers=num_of_naive_net_layers,
                       naive_net_output_dims=naive_net_output_dims)
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, training_loader, validation_loader)

    return trainer.callback_metrics["val_epoch/mse_loss"].item()


def objective_naive_net_with_SC(trial: optuna.Trial, training_loader, validation_loader, save_dir: str) -> float:
    # training params
    max_epochs = 100
    save_top_k = 5

    # We optimize the number of layers and hidden units in each layer of naive net as well as that of SC feature extraction net.
    num_of_extract_SC_feat_layers = trial.suggest_int("num_of_extract_SC_feat_layers", 1, 5)
    extract_SC_feat_output_dims = [
        trial.suggest_int(f"extract_SC_feat_output_dim_l{i}", 4, 256, log=True)
        for i in range(num_of_extract_SC_feat_layers)
    ]
    num_of_naive_net_layers = trial.suggest_int("num_of_naive_net_layers", 1, 5)
    naive_net_output_dims = [
        trial.suggest_int(f"naive_net_output_dim_l{i}", 4, 256, log=True) for i in range(num_of_naive_net_layers)
    ]

    extract_SC_feat_net = ExtractScFeatMLP(extract_SC_feat_output_dims)
    model = NaiveNet(naive_net_output_dims, extract_SC_feat_net=extract_SC_feat_net)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, save_top_k),
        #   limit_train_batches=4,
        #   limit_val_batches=4,
        #   limit_test_batches=4,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=LOG_INTERVAL,
        enable_progress_bar=False)

    hyperparams = dict(num_of_extract_SC_feat_layers=num_of_extract_SC_feat_layers,
                       extract_SC_feat_output_dims=extract_SC_feat_output_dims,
                       num_of_naive_net_layers=num_of_naive_net_layers,
                       naive_net_output_dims=naive_net_output_dims)
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, training_loader, validation_loader)

    return trainer.callback_metrics["val_epoch/mse_loss"].item()
