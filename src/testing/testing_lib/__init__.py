import sys
import os
import torch

from src.utils.init_utils import seed_all, set_gpu_device, load_dataset, load_split

from src.basic.constants import PATH_TO_DATASET, PATH_TO_TESTING_REPORT, PATH_TO_TRAINING_LOG, PATH_TO_FIGURES, MANUAL_SEED

from src.models.extract_SC_feat import ExtractScFeatMLP
from src.models.naive_net import NaiveNet
from src.models.gnn.gcn import Gcn

from src.utils.test_utils import check_all_good_params_costs_for_models_in, get_top_k_prediction, gnn_get_top_k_prediction, plot_top_k_distribution, all_costs_prediction_vs_actual_cost, pred_and_actual_cost_corr_dist
from src.utils.SC_utils import df_to_tensor, get_path_to_group
from src.utils.pMFM_utils import forward_simulation


def load_naive_net_with_SC():
    path_to_model = os.path.join(
        PATH_TO_TRAINING_LOG,
        'basic_models/naive_net/with_SC/lightning_logs/version_0/checkpoints/epoch=91-step=204884-val_mse=0.0186491.ckpt'
    )
    extract_SC_feat_output_dims = [147, 9]
    naive_net_output_dims = [29, 48, 94, 32, 13]
    extract_SC_feat_net = ExtractScFeatMLP(extract_SC_feat_output_dims)
    model = NaiveNet.load_from_checkpoint(path_to_model,
                                          output_dims=naive_net_output_dims,
                                          extract_SC_feat_net=extract_SC_feat_net)
    model.to(device)
    return model


def load_naive_net_no_SC():
    path_to_model = os.path.join(
        PATH_TO_TRAINING_LOG,
        'basic_models/naive_net/no_SC/lightning_logs/version_74/checkpoints/epoch=96-step=216019-val_mse=0.0106791.ckpt'
    )
    naive_net_output_dims = [24, 189, 158, 15, 10, 16, 58, 27]
    model = NaiveNet.load_from_checkpoint(path_to_model, output_dims=naive_net_output_dims)
    model.to(device)
    return model


def load_gcn_with_mlp():
    path_to_model = os.path.join(
        PATH_TO_TRAINING_LOG,
        'gnn/gcn_with_mlp/lightning_logs/version_19/checkpoints/epoch=36-step=82399-val_mse=0.0379823.ckpt')
    gcn_output_dims = [15]
    mlp_output_dims = [135, 41, 4, 15]
    model = Gcn.load_from_checkpoint(path_to_model, gcn_output_dims=gcn_output_dims, mlp_output_dims=mlp_output_dims)
    model.to(device)
    return model


seed_all()

device, run_on_gpu = set_gpu_device()
