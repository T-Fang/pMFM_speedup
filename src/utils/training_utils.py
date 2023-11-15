import torch
import os
import pickle
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch_geometric.loader import DataLoader as GnnDataLoader

from src.basic.constants import MANUAL_SEED, PATH_TO_TRAINING_LOG
from src.models.gnn.gnn_param_dataset import GnnParamDataset
from src.utils.init_utils import load_split


def get_trainer_callbacks(trial, save_top_k: int):
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_epoch/mse_loss")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(save_top_k=save_top_k,
                                          monitor="val_epoch/mse_loss",
                                          mode="min",
                                          save_last=True,
                                          filename="epoch={epoch}-step={step}-val_mse={val_epoch/mse_loss:.7f}",
                                          auto_insert_metric_name=False)
    checkpoint_callback.CHECKPOINT_NAME_LAST = "epoch={epoch}-step={step}-last"

    return [checkpoint_callback, lr_monitor, optuna_pruning]


def print_best_trial(study: optuna.Trial):
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Trial number: {trial.number}")
    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def dataset2loader(dataset, shuffle: bool):
    BATCH_SIZE = 256
    NUM_WORKERS = 0
    PIN_MEMORY = True

    # Create data loaders for our datasets; should shuffle for training, but not for validation
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=shuffle,
                                         num_workers=NUM_WORKERS,
                                         pin_memory=PIN_MEMORY)
    return loader


def tune(objective,
         train_ds=None,
         validation_ds=None,
         n_trials=100,
         timeout=36000,
         save_dir=PATH_TO_TRAINING_LOG,
         use_coef=False):

    if train_ds is None or validation_ds is None:
        train_ds, validation_ds = load_split('train', use_coef=use_coef), load_split('validation', use_coef=use_coef)

    training_loader = dataset2loader(train_ds, shuffle=True)
    validation_loader = dataset2loader(validation_ds, shuffle=False)

    study_file_path = os.path.join(save_dir, "study.pkl")
    if os.path.isfile(study_file_path):
        study = pickle.load(open(study_file_path, "rb"))
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        sampler = TPESampler(seed=MANUAL_SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda trial: objective(trial, training_loader, validation_loader, save_dir),
                   n_trials=n_trials,
                   timeout=timeout)

    print_best_trial(study)
    pickle.dump(study, open(study_file_path, "wb"))


def dataset2loader_gnn(dataset, shuffle: bool):
    BATCH_SIZE = 256
    NUM_WORKERS = 0
    PIN_MEMORY = True

    # Create data loaders for our gnn datasets; should shuffle for training, but not for validation
    loader = GnnDataLoader(dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=shuffle,
                           num_workers=NUM_WORKERS,
                           pin_memory=PIN_MEMORY)
    return loader


def tune_gnn(objective, train_ds=None, validation_ds=None, n_trials=100, timeout=36000, save_dir=PATH_TO_TRAINING_LOG):
    if train_ds is None or validation_ds is None:
        train_ds, validation_ds = GnnParamDataset('train'), GnnParamDataset('validation')

    training_loader = dataset2loader_gnn(train_ds, shuffle=True)
    validation_loader = dataset2loader_gnn(validation_ds, shuffle=False)

    study_file_path = os.path.join(save_dir, "study.pkl")
    if os.path.isfile(study_file_path):
        study = pickle.load(open(study_file_path, "rb"))
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        sampler = TPESampler(seed=MANUAL_SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda trial: objective(trial, training_loader, validation_loader, save_dir),
                   n_trials=n_trials,
                   timeout=timeout)

    print_best_trial(study)
    pickle.dump(study, open(study_file_path, "wb"))
