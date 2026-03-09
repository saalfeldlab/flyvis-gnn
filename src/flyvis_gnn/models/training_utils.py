"""Shared utilities for FlyVis training and testing.

Extracted from graph_trainer.py to eliminate duplication of data loading,
model construction, checkpoint management, and optimizer setup across
data_train_flyvis, data_test_flyvis, and data_test_flyvis_special.
"""

import glob
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from flyvis_gnn.models.registry import create_model
from flyvis_gnn.models.utils import set_trainable_parameters
from flyvis_gnn.utils import graphs_data_path, migrate_state_dict, sort_key
from flyvis_gnn.zarr_io import load_raw_array, load_simulation_data


def determine_load_fields(config):
    """Determine which NeuronTimeSeries fields to load based on config.

    Returns:
        list of field name strings for load_simulation_data().
    """
    model_config = config.graph_model
    sim = config.simulation
    fields = ['voltage', 'stimulus', 'neuron_type']
    if 'visual' in model_config.field_type or 'test' in model_config.field_type:
        fields.append('pos')
    if sim.calcium_type != 'none':
        fields.append('calcium')
    if sim.measurement_noise_level > 0:
        fields.append('noise')
    return fields


def load_flyvis_data(dataset_name, split='train', fields=None, device=None,
                     training_selected_neurons=False, selected_neuron_ids=None,
                     measurement_noise_level=0.0):
    """Load NeuronTimeSeries + derivative targets for a given split.

    Handles backwards compatibility: falls back to x_list_0 / y_list_0
    if the requested split (x_list_train / x_list_test) does not exist.

    Args:
        dataset_name: dataset identifier (e.g. 'fly/flyvis_noise_005')
        split: 'train' or 'test'
        fields: list of field names to load (from determine_load_fields)
        device: torch device to move data to
        training_selected_neurons: if True, subset neurons
        selected_neuron_ids: list of neuron indices to keep
        measurement_noise_level: if > 0, load noisy_y_list instead of y_list

    Returns:
        x_ts: NeuronTimeSeries on device
        y_ts: numpy array of derivative targets, shape (T, N, 1)
        type_list: (N, 1) float tensor of neuron type labels
    """
    split_name = f'x_list_{split}'
    path = graphs_data_path(dataset_name, split_name)

    # Choose derivative target: noisy or clean
    y_prefix = 'noisy_y_list' if measurement_noise_level > 0 else 'y_list'

    if os.path.exists(path):
        x_ts = load_simulation_data(path, fields=fields).to(device)
        y_ts = load_raw_array(graphs_data_path(dataset_name, f'{y_prefix}_{split}'))
    else:
        print(f"warning: {split_name} not found, falling back to x_list_0")
        x_ts = load_simulation_data(
            graphs_data_path(dataset_name, 'x_list_0'), fields=fields
        ).to(device)
        y_ts = load_raw_array(graphs_data_path(dataset_name, 'y_list_0'))

    # Extract type_list, then construct index (not loaded from disk)
    type_list = x_ts.neuron_type.float().unsqueeze(-1)
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)

    if training_selected_neurons and selected_neuron_ids is not None:
        selected = np.array(selected_neuron_ids).astype(int)
        x_ts = x_ts.subset_neurons(selected)
        y_ts = y_ts[:, selected, :]
        type_list = type_list[selected]

    return x_ts, y_ts, type_list


def build_model(config, device, checkpoint_path=None, reset_epoch=False):
    """Create a FlyVisGNN model and optionally load a checkpoint.

    Args:
        config: NeuralGraphConfig
        device: torch device
        checkpoint_path: path to .pt checkpoint file (or None)

    Returns:
        model: FlyVisGNN on device
        start_epoch: int, 0 unless resumed from a checkpoint with epoch in filename
    """
    model_config = config.graph_model
    model = create_model(
        model_config.signal_model_name,
        aggr_type=model_config.aggr_type,
        config=config, device=device,
    ).to(device)

    # Resolve relative ./log/... paths against data_root
    if checkpoint_path and not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        from flyvis_gnn.utils import get_data_root
        resolved = os.path.join(get_data_root(), checkpoint_path.lstrip('./'))
        if os.path.exists(resolved):
            checkpoint_path = resolved

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'loading state_dict from {checkpoint_path} ...')
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        migrate_state_dict(state_dict)
        model.load_state_dict(state_dict['model_state_dict'])

        # Try to extract epoch from filename (e.g. best_model_with_1_graphs_5.pt → epoch 5)
        basename = os.path.basename(checkpoint_path)
        name_no_ext = basename.replace('.pt', '')
        parts = name_no_ext.split('_')
        try:
            start_epoch = int(parts[-1])
        except (ValueError, IndexError):
            pass
        if reset_epoch:
            start_epoch = 0
        print(f'state_dict loaded, start_epoch={start_epoch}')
    else:
        if checkpoint_path:
            print(f'checkpoint not found: {checkpoint_path} — using freshly initialized model')
        else:
            print('no state_dict loaded — using freshly initialized model')

    return model, start_epoch


def resolve_checkpoint_path(config, best_model):
    """Resolve a best_model specifier to an absolute checkpoint path.

    Handles:
        None / '' / 'None' → pretrained_model from config (or None)
        'best' → glob for latest checkpoint
        numeric string (e.g. '5') → specific epoch checkpoint

    Returns:
        str path or None
    """
    tc = config.training
    log_dir = os.path.dirname(os.path.dirname(
        graphs_data_path(config.dataset, '')))  # not ideal but matches existing pattern
    # Use the actual log_path utility
    from flyvis_gnn.utils import log_path as _log_path
    log_dir = _log_path(config.config_file)

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
        if not files:
            return None
        files.sort(key=sort_key)
        filename = files[-1].split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        return f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{filename}.pt"
    elif best_model and best_model != 'None' and best_model != '':
        return f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt"
    elif tc.pretrained_model != '':
        return tc.pretrained_model
    return None


def build_optimizer(model, config):
    """Build Adam optimizer with per-parameter-group learning rates.

    Wraps set_trainable_parameters with the learning rates from config.

    Returns:
        optimizer: torch.optim.Adam
        n_total_params: int
    """
    tc = config.training
    lr = tc.learning_rate_start
    lr_update = tc.learning_rate_update_start if tc.learning_rate_update_start != 0 else lr
    lr_embedding = tc.learning_rate_embedding_start
    lr_W = tc.learning_rate_W_start
    learning_rate_NNR = tc.learning_rate_NNR
    learning_rate_NNR_f = tc.learning_rate_NNR_f

    return set_trainable_parameters(
        model=model, lr_embedding=lr_embedding, lr=lr,
        lr_update=lr_update, lr_W=lr_W,
        learning_rate_NNR=learning_rate_NNR,
        learning_rate_NNR_f=learning_rate_NNR_f,
    )


def build_lr_scheduler(optimizer, config):
    """Build LR scheduler from config.

    Supports:
        'none': constant LR (no-op scheduler)
        'cosine_warm_restarts': CosineAnnealingWarmRestarts per iteration
        'linear_warmup_cosine': linear warmup then cosine decay

    New config fields (all with backwards-compatible defaults):
        training.lr_scheduler: str = 'none'
        training.lr_scheduler_T0: int = 1000
        training.lr_scheduler_T_mult: int = 2
        training.lr_scheduler_eta_min_ratio: float = 0.01
        training.lr_scheduler_warmup_iters: int = 100

    Returns:
        LR scheduler instance
    """
    tc = config.training
    scheduler_type = getattr(tc, 'lr_scheduler', 'none')

    if scheduler_type == 'none':
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    elif scheduler_type == 'cosine_warm_restarts':
        T_0 = getattr(tc, 'lr_scheduler_T0', 1000)
        T_mult = getattr(tc, 'lr_scheduler_T_mult', 2)
        eta_min_ratio = getattr(tc, 'lr_scheduler_eta_min_ratio', 0.01)

        # Compute per-group eta_min from initial lr
        eta_min = min(pg['lr'] for pg in optimizer.param_groups) * eta_min_ratio

        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    elif scheduler_type == 'linear_warmup_cosine':
        warmup_iters = getattr(tc, 'lr_scheduler_warmup_iters', 100)
        T_0 = getattr(tc, 'lr_scheduler_T0', 1000)
        T_mult = getattr(tc, 'lr_scheduler_T_mult', 2)
        eta_min_ratio = getattr(tc, 'lr_scheduler_eta_min_ratio', 0.01)
        eta_min = min(pg['lr'] for pg in optimizer.param_groups) * eta_min_ratio

        def lr_lambda(step):
            if step < warmup_iters:
                return max(step / max(warmup_iters, 1), 1e-6)
            return 1.0

        warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        return torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])

    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_type}")


def save_checkpoint(model, optimizer, path):
    """Save model + optimizer state dict to path."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
