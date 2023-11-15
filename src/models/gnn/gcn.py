from typing import List

import torch
from torch.nn import Linear, ReLU, BatchNorm1d

import optuna

from pytorch_lightning import Trainer, loggers as pl_loggers
from torch_geometric.nn import GCNConv, Sequential

from src.basic.pl_module import PlModule
from src.basic.cost_type import CostType
from src.basic.constants import NUM_REGION, LOG_INTERVAL
from src.utils.gnn_utils import RebatchNodeFeat
from src.utils.training_utils import get_trainer_callbacks

NUM_NODE_HIDDEN_FEAT = 3

INPUT_DIM_TO_FC2 = 200
INPUT_DIM_TO_FC3 = 64


class Gcn(PlModule):

    def __init__(self, gcn_output_dims: List[int], mlp_output_dims: List[int]):
        super().__init__()

        self.output_dims = gcn_output_dims

        input_dim = 3

        modules = []
        for i, output_dim in enumerate(gcn_output_dims, 0):
            conv_layer = GCNConv(input_dim, output_dim, add_self_loops=False)
            if i == 0:
                # * if later Batch Norm is added, can set `bias=False` for GCNConv
                modules.append((conv_layer, f"x, edge_index, edge_weight -> x{i}"))
            else:
                modules.append((conv_layer, f"x{i-1}a, edge_index, edge_weight -> x{i}"))
            modules.append((ReLU(), f"x{i} -> x{i}a"))

            input_dim = output_dim

        # Concatenate node features within one graph
        modules.append((RebatchNodeFeat(), f"x{len(gcn_output_dims)-1}a -> x_rebatched"))
        # x_rebatched has shape (batch_size, NUM_REGION * input_dim)

        # Add an MLP at the end
        for i, output_dim in enumerate(mlp_output_dims, 0):
            if i == 0:
                modules.append((Linear(NUM_REGION * input_dim, output_dim), f"x_rebatched -> x_mlp{i}"))
            else:
                modules.append((Linear(input_dim, output_dim), f"x_mlp{i-1}bn -> x_mlp{i}"))
            modules.append((ReLU(), f"x_mlp{i} -> x_mlp{i}a"))
            modules.append((BatchNorm1d(output_dim), f"x_mlp{i}a -> x_mlp{i}bn"))

            input_dim = output_dim

        # Add the output/prediction layer
        modules.append((Linear(input_dim, CostType.num_of_cost_type()), f"x_mlp{len(mlp_output_dims)-1}bn -> y_hat"))

        self.model = Sequential("x, edge_index, edge_weight", modules)

    def forward(self, batch):
        """
        generate sequential features from a large graph formed by graphs in a batch

        Args:
            batch (torch_geometric.data.Batch): Data describing the large batch graph, inherits from torch_geometric.data.Data and contains an additional attribute called `batch` # noqa: E501
        """

        # Note that the batch_indices here is used to assign nodes to their corresponding graphs
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_weight

        y_hat = self.model(x, edge_index, edge_weight)
        return y_hat

    def get_y_hat_and_y(self, batch):
        y_hat = self(batch)
        y = batch.y
        return y_hat, y


def objective_gcn(trial: optuna.Trial, training_loader, validation_loader, save_dir: str) -> float:
    # training params
    max_epochs = 100
    save_top_k = 5

    # We optimize the number of layers and hidden units in each layer of naive net.
    num_of_gcn_layers = trial.suggest_int("num_of_gcn_layers", 1, 3)
    gcn_output_dims = [trial.suggest_int(f"gcn_output_dim_{i}", 4, 128, log=True) for i in range(num_of_gcn_layers)]
    num_of_mlp_layers = trial.suggest_int("num_of_mlp_layers", 1, 10)
    mlp_output_dims = [trial.suggest_int(f"mlp_output_dim_{i}", 4, 512, log=True) for i in range(num_of_mlp_layers)]
    model = Gcn(gcn_output_dims, mlp_output_dims)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, save_top_k),
        # limit_train_batches=4,
        # limit_val_batches=4,
        # limit_test_batches=4,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=LOG_INTERVAL,
        enable_progress_bar=False)

    hyperparams = dict(num_of_gcn_layers=num_of_gcn_layers,
                       gcn_output_dims=gcn_output_dims,
                       num_of_mlp_layers=num_of_mlp_layers,
                       mlp_output_dims=mlp_output_dims)
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(model, training_loader, validation_loader)

    return trainer.callback_metrics["val_epoch/mse_loss"].item()
