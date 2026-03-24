"""Test functions for FlyVis GNN models.

Extracted from graph_trainer.py to reduce file size.
Contains:
- data_test_flyvis: standard test with 1-step + rollout evaluation
- data_test_flyvis_special: ablation/modification test via ODE regeneration
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from flyvis_gnn.figure_style import dark_style
from flyvis_gnn.generators.graph_data_generator import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
from flyvis_gnn.generators.ode_params import FlyVisODEParams
from flyvis_gnn.generators.utils import generate_compressed_video_mp4
from flyvis_gnn.log import get_logger
from flyvis_gnn.metrics import INDEX_TO_NAME
from flyvis_gnn.models.Neural_ode_wrapper_FlyVis import integrate_neural_ode_FlyVis
from flyvis_gnn.models.registry import create_model
from flyvis_gnn.models.utils import _batch_frames
from flyvis_gnn.neuron_state import NeuronState
from flyvis_gnn.plot import plot_spatial_activity_grid, plot_weight_comparison
from flyvis_gnn.utils import (
    compute_trace_metrics,
    get_datavis_root_dir,
    get_equidistant_points,
    graphs_data_path,
    log_path,
    migrate_state_dict,
    sort_key,
    to_numpy,
)
from flyvis_gnn.zarr_io import load_raw_array, load_simulation_data

try:
    from flyvis_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset
except ImportError:
    AugmentedVideoDataset = None
    CombinedVideoDataset = None

logger = get_logger(__name__)


def data_test_flyvis(config, best_model=None, device=None, log_file=None, test_config=None):
    """Test using pre-generated test data (x_list_test / y_list_test).

    Loads the held-out test split, runs the trained model on every frame,
    and reports per-neuron RMSE, Pearson r, R², and FEVE.

    Args:
        config: model config (model + log dir come from here)
        test_config: optional second config for cross-dataset evaluation
                     (test data loaded from test_config.dataset)
    """

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    log_dir = log_path(config.config_file)

    # Determine test dataset: test_config > tc.test_dataset > config.dataset
    if test_config is not None:
        test_ds = test_config.dataset
        logger.info(f'cross-dataset test: model from {config.dataset}, test data from {test_ds}')
    elif tc.test_dataset:
        test_ds = tc.test_dataset
    else:
        test_ds = config.dataset

    # Suffix for output files when testing on a different dataset
    if test_ds != config.dataset:
        test_ds_short = test_ds.replace('flyvis_', '').replace('fly/', '')
        test_suffix = f'_on_{test_ds_short}'
    else:
        test_suffix = ''

    # Determine which fields to load
    load_fields = ['voltage', 'stimulus', 'neuron_type']
    has_visual_field = 'visual' in model_config.field_type
    if has_visual_field or 'test' in model_config.field_type:
        load_fields.append('pos')
    if sim.calcium_type != 'none':
        load_fields.append('calcium')

    # Load test data (fall back to x_list_0 for backwards compatibility)
    test_path = graphs_data_path(test_ds, 'x_list_test')
    if os.path.exists(test_path):
        x_ts = load_simulation_data(test_path, fields=load_fields).to(device)
        y_ts = load_raw_array(graphs_data_path(test_ds, 'y_list_test'))
    else:
        logger.warning("x_list_test not found, falling back to x_list_0")
        x_ts = load_simulation_data(
            graphs_data_path(test_ds, 'x_list_0'), fields=load_fields
        ).to(device)
        y_ts = load_raw_array(graphs_data_path(test_ds, 'y_list_0'))

    # Extract type_list and set up index
    type_list = x_ts.neuron_type.float().unsqueeze(-1)
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)

    if tc.training_selected_neurons:
        selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
        x_ts = x_ts.subset_neurons(selected_neuron_ids)
        y_ts = y_ts[:, selected_neuron_ids, :]
        type_list = type_list[selected_neuron_ids]

    n_neurons = x_ts.n_neurons
    n_frames = x_ts.n_frames
    config.simulation.n_neurons = n_neurons
    logger.info(f'\033[94mtest dataset: {test_ds}\033[0m, {n_frames} frames, {n_neurons} neurons')

    # Adjust n_edges to match saved edge_index (edge removal changes the count)
    ode_params = FlyVisODEParams.load(graphs_data_path(config.dataset), device='cpu')
    edges_for_size = ode_params.edge_index
    actual_n_edges = edges_for_size.shape[1]
    expected_total = sim.n_edges + sim.n_extra_null_edges
    if actual_n_edges == expected_total and sim.n_extra_null_edges > 0:
        logger.info(f'null edges in data: {sim.n_edges} base + {sim.n_extra_null_edges} null = {actual_n_edges}')
    elif actual_n_edges != sim.n_edges:
        logger.info(f'n_edges mismatch: config={sim.n_edges}, actual={actual_n_edges} — using actual')
        config.simulation.n_edges = actual_n_edges

    # Create and load model
    logger.info('creating model ...')
    model = create_model(
        model_config.signal_model_name,
        aggr_type=model_config.aggr_type, config=config, device=device,
    )
    model = model.to(device)

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        logger.info(f'best model: {best_model}')
    netname = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt"
    logger.info(f'loading {netname} ...')
    state_dict = torch.load(netname, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    model.load_state_dict(state_dict['model_state_dict'])

    # Load INR model if visual field is learned
    if has_visual_field and hasattr(model, 'NNR_f'):
        # Extract epoch from best_model to find matching INR checkpoint
        epoch_str = best_model.split('_')[0] if best_model else '0'
        inr_path = os.path.join(log_dir, 'models', f'inr_stimulus_{epoch_str}.pt')
        if os.path.exists(inr_path):
            model.NNR_f.load_state_dict(torch.load(inr_path, map_location=device, weights_only=False))
            logger.info(f'loaded INR from {inr_path}')
        else:
            logger.warning(f'INR checkpoint not found at {inr_path}')

    model.eval()

    # Apply ablation mask if test dataset has one
    test_ds_for_mask = test_ds if test_config is not None else config.dataset
    mask_path = graphs_data_path(test_ds_for_mask, 'ablation_mask.pt')
    if os.path.exists(mask_path):
        ablation_mask = torch.load(mask_path, map_location=device, weights_only=False)
        with torch.no_grad():
            model.W[~ablation_mask] = 0
        logger.info(f'applied ablation mask: {(~ablation_mask).sum().item()} edges zeroed in model.W')

    # When visual field is learned, use training data (INR was fit to it)
    if has_visual_field:
        train_path = graphs_data_path(config.dataset, 'x_list_train')
        if os.path.exists(train_path):
            x_ts_train = load_simulation_data(train_path, fields=load_fields).to(device)
            y_ts_train = load_raw_array(graphs_data_path(config.dataset, 'y_list_train'))
            x_ts_train.neuron_type = None
            x_ts_train.index = torch.arange(x_ts_train.n_neurons, dtype=torch.long, device=device)
            if tc.training_selected_neurons:
                x_ts_train = x_ts_train.subset_neurons(selected_neuron_ids)
                y_ts_train = y_ts_train[:, selected_neuron_ids, :]
            n_eval_frames = min(n_frames, x_ts_train.n_frames)
            logger.info(f'visual field learned: evaluating on training data ({x_ts_train.n_frames} frames available, using {n_eval_frames})')
            x_ts_eval = x_ts_train
            y_ts_eval = y_ts_train
        else:
            logger.warning('x_list_train not found, falling back to test data')
            x_ts_eval = x_ts
            y_ts_eval = y_ts
            n_eval_frames = n_frames
    else:
        x_ts_eval = x_ts
        y_ts_eval = y_ts
        n_eval_frames = n_frames

    # Load edges from training dataset (model was trained on these edges)
    edges = ode_params.edge_index.to(device)
    ids = np.arange(n_neurons)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    # Run model on all frames (one-step prediction)
    logger.info(f'\033[93mone-step prediction on {n_eval_frames} frames ...\033[0m')
    all_pred = []
    all_true = []

    with torch.no_grad():
        for k in range(n_eval_frames - 1):
            x = x_ts_eval.frame(k)
            y = torch.tensor(y_ts_eval[k], device=device)

            if torch.isnan(x.voltage).any() or torch.isnan(y).any():
                continue

            if has_visual_field:
                visual_input = model.forward_visual(x, k)
                x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                x.stimulus[model.n_input_neurons:] = 0

            if 'MLP' in model_config.signal_model_name:
                batched_state, _ = _batch_frames([x], edges)
                x_packed = batched_state.to_packed()
                pred = model(x_packed, data_id=data_id, return_all=False)
            else:
                batched_state, batched_edges = _batch_frames([x], edges)
                pred, _, _ = model(
                    batched_state, batched_edges,
                    data_id=data_id, return_all=True,
                )

            all_pred.append(to_numpy(pred.squeeze()))
            all_true.append(to_numpy(y.squeeze()))

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    # Compute per-neuron metrics: transpose to (n_neurons, n_frames)
    rmse, pearson, feve, r2 = compute_trace_metrics(
        all_true.T, all_pred.T, label="test"
    )

    # Save results
    results_path = os.path.join(log_dir, f'results_test{test_suffix}.log')
    with open(results_path, 'w') as f:
        f.write(f'test_dataset: {test_ds}\n')
        f.write(f'n_frames: {len(all_pred)}\n')
        f.write(f'n_neurons: {n_neurons}\n')
        f.write(f'model: {netname}\n')
        f.write(f'Pearson r: {np.nanmean(pearson):.3f} +/- {np.nanstd(pearson):.3f}\n')
        f.write(f'RMSE: {np.mean(rmse):.4f} +/- {np.std(rmse):.4f}\n')
    logger.debug(f'results saved to {results_path}')

    if log_file:
        log_file.write('\n--- Pre-generated test results ---\n')
        log_file.write(f'test_dataset: {test_ds}\n')
        log_file.write(f'Pearson r: {np.nanmean(pearson):.3f} +/- {np.nanstd(pearson):.3f}\n')
        log_file.write(f'RMSE: {np.mean(rmse):.4f} +/- {np.std(rmse):.4f}\n')

    # --- Rollout evaluation ---
    # Start from initial voltages at t=0, predict autoregressively
    logger.info('\033[93mrunning rollout evaluation ...\033[0m')
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    x = x_ts_eval.frame(0)

    h_state = None
    c_state = None

    rollout_pred_list = []
    rollout_true_list = []
    rollout_stim_list = []
    stimuli_true_list = []   # true stimulus (input neurons only)
    stimuli_pred_list = []   # SIREN predicted stimulus (input neurons only)

    with torch.no_grad():
        for k in trange(n_eval_frames - 1, ncols=100, desc="rollout"):
            # Collect state before integration
            rollout_pred_list.append(to_numpy(x.voltage))
            rollout_true_list.append(to_numpy(x_ts_eval.frame(k).voltage))

            # Set stimulus from rollout data
            x.stimulus = x_ts_eval.frame(k).stimulus.clone()
            rollout_stim_list.append(to_numpy(x.stimulus))

            if has_visual_field:
                stimuli_true_list.append(to_numpy(x.stimulus[:model.n_input_neurons]))
                visual_input = model.forward_visual(x, k)
                stimuli_pred_list.append(to_numpy(visual_input.squeeze(-1)))
                x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                x.stimulus[model.n_input_neurons:] = 0

            # Model prediction
            if 'RNN' in model_config.signal_model_name:
                y, h_state = model(x.to_packed(), h=h_state, return_all=True)
            elif 'LSTM' in model_config.signal_model_name:
                y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
            elif 'MLP_ODE' in model_config.signal_model_name:
                v = x.voltage.unsqueeze(-1)
                if tc.training_selected_neurons:
                    I = x.stimulus.unsqueeze(-1)
                else:
                    I = x.stimulus[:sim.n_input_neurons].unsqueeze(-1)
                y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v
            elif 'MLP' in model_config.signal_model_name:
                y = model(x.to_packed(), data_id=data_id, return_all=False)
            elif hasattr(tc, 'neural_ODE_training') and tc.neural_ODE_training:
                v0 = x.voltage.flatten()
                v_final, _ = integrate_neural_ode_FlyVis(
                    model=model, v0=v0, x_template=x,
                    edge_index=edges, data_id=data_id,
                    time_steps=1, delta_t=sim.delta_t,
                    neurons_per_sample=n_neurons, batch_size=1,
                    has_visual_field=has_visual_field,
                    x_ts=None, device=device,
                    k_batch=torch.tensor([k], device=device),
                    ode_method=tc.ode_method,
                    rtol=tc.ode_rtol, atol=tc.ode_atol,
                    adjoint=False, noise_level=0.0
                )
                y = (v_final.view(-1, 1) - x.voltage.unsqueeze(-1)) / sim.delta_t
            else:
                y = model(x, edges, data_id=data_id, return_all=False)

            # Integration step
            if 'MLP_ODE' in model_config.signal_model_name:
                x.voltage = x.voltage + y.squeeze(-1)
            else:
                x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)

            # Guard against NaN / divergence from a poorly trained model
            if torch.isnan(x.voltage).any() or torch.isinf(x.voltage).any():
                logger.error(f"rollout diverged at frame {k} (NaN/Inf in voltage) — aborting")
                break
            x.voltage = torch.clamp(x.voltage, min=-100.0, max=100.0)

            # Calcium dynamics
            if sim.calcium_type == "leaky":
                if sim.calcium_activation == "softplus":
                    u = torch.nn.functional.softplus(x.voltage)
                elif sim.calcium_activation == "relu":
                    u = torch.nn.functional.relu(x.voltage)
                elif sim.calcium_activation == "tanh":
                    u = torch.tanh(x.voltage)
                elif sim.calcium_activation == "identity":
                    u = x.voltage.clone()
                x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + u)
                x.calcium = torch.clamp(x.calcium, min=0.0)
                x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta

    rollout_pred_arr = np.array(rollout_pred_list)   # (n_frames-1, n_neurons)
    rollout_true_arr = np.array(rollout_true_list)   # (n_frames-1, n_neurons)
    rollout_stim_arr = np.array(rollout_stim_list)   # (n_frames-1, n_neurons)

    activity_pred = rollout_pred_arr.T   # (n_neurons, n_frames-1)
    activity_true = rollout_true_arr.T   # (n_neurons, n_frames-1)
    stimulus_arr = rollout_stim_arr.T    # (n_neurons, n_frames-1)

    # Compute stimuli_R2: SIREN output vs true stimulus (with linear correction ax+b)
    stimuli_R2 = None
    if has_visual_field and stimuli_true_list:
        stim_true_2d = np.array(stimuli_true_list)   # (n_frames, n_input_neurons)
        stim_pred_2d = np.array(stimuli_pred_list)   # (n_frames, n_input_neurons)
        # Global linear fit: true = a * pred + b
        pred_flat = stim_pred_2d.ravel()
        true_flat = stim_true_2d.ravel()
        A_fit = np.vstack([pred_flat, np.ones(len(pred_flat))]).T
        a_coeff, b_coeff = np.linalg.lstsq(A_fit, true_flat, rcond=None)[0]
        pred_corrected = a_coeff * stim_pred_2d + b_coeff
        ss_res = np.sum((stim_true_2d - pred_corrected) ** 2)
        ss_tot = np.sum((stim_true_2d - np.mean(stim_true_2d)) ** 2)
        stimuli_R2 = float(1 - ss_res / (ss_tot + 1e-16))
        logger.info(f'stimuli_R2 (corrected a={a_coeff:.4f} b={b_coeff:.4f}): {stimuli_R2:.4f}')

        # Generate stimuli GT vs Pred video
        if hasattr(x_ts_eval.frame(0), 'pos') and x_ts_eval.frame(0).pos is not None:
            from flyvis_gnn.models.graph_trainer_inr import _generate_inr_video
            pos_input = to_numpy(x_ts_eval.frame(0).pos[:model.n_input_neurons])
            results_dir = os.path.join(log_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            _generate_inr_video(
                gt_np=stim_true_2d,
                predict_frame_fn=lambda k: stim_pred_2d[k],
                pos_np=pos_input,
                field_name='stimulus',
                output_folder=results_dir,
                n_frames=stim_true_2d.shape[0],
            )

    # Compute rollout metrics
    rmse_ro, pearson_ro, feve_ro, r2_ro = compute_trace_metrics(
        activity_true, activity_pred, label="rollout"
    )

    # Save rollout metrics
    rollout_log_path = os.path.join(log_dir, f'results_rollout{test_suffix}.log')
    with open(rollout_log_path, 'w') as f:
        f.write("Rollout Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"RMSE: {np.mean(rmse_ro):.4f} +/- {np.std(rmse_ro):.4f}\n")
        f.write(f"Pearson r: {np.nanmean(pearson_ro):.3f} +/- {np.nanstd(pearson_ro):.3f}\n")
        # f.write(f"R2: {np.nanmean(r2_ro):.3f} +/- {np.nanstd(r2_ro):.3f}\n")
        # f.write(f"FEVE: {np.mean(feve_ro):.3f} +/- {np.std(feve_ro):.3f}\n")
        f.write(f"\nNumber of neurons evaluated: {n_neurons}\n")
        f.write(f"Frames evaluated: 0 to {n_eval_frames - 1}\n")
        if has_visual_field:
            f.write("Rollout data source: training (INR learned on training data)\n")
        if stimuli_R2 is not None:
            f.write(f"stimuli_R2: {stimuli_R2:.4f}\n")
    logger.debug(f'rollout metrics saved to {rollout_log_path}')

    if log_file:
        log_file.write('\n--- Rollout results ---\n')
        log_file.write(f'RMSE: {np.mean(rmse_ro):.4f} +/- {np.std(rmse_ro):.4f}\n')
        log_file.write(f'Pearson r: {np.nanmean(pearson_ro):.3f} +/- {np.nanstd(pearson_ro):.3f}\n')
        # log_file.write(f'R2: {np.nanmean(r2_ro):.3f} +/- {np.nanstd(r2_ro):.3f}\n')
        # log_file.write(f'FEVE: {np.mean(feve_ro):.3f} +/- {np.std(feve_ro):.3f}\n')
        if stimuli_R2 is not None:
            log_file.write(f'stimuli_R2: {stimuli_R2:.4f}\n')

    # --- Rollout trace plots ---
    neuron_types = to_numpy(type_list).astype(int).squeeze()
    n_neuron_types = sim.n_neuron_types
    index_to_name = INDEX_TO_NAME

    start_frame = 0
    end_frame = activity_true.shape[1]

    filename_ = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'test'

    for fig_name, selected_types in [
        ("selected", [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]),
        ("all", np.arange(0, n_neuron_types)),
    ]:
        neuron_indices = []
        for stype in selected_types:
            indices = np.where(neuron_types == stype)[0]
            if len(indices) > 0:
                neuron_indices.append(indices[0])

        if not neuron_indices:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        true_slice = activity_true[neuron_indices, start_frame:end_frame]
        stim_slice = stimulus_arr[neuron_indices, start_frame:end_frame]
        pred_slice = activity_pred[neuron_indices, start_frame:end_frame]
        step_v = 2.5
        lw = 2

        name_fontsize = 10 if len(selected_types) > 50 else 18

        # ground truth (green, thick)
        baselines = {}
        for i in range(len(neuron_indices)):
            baseline = np.mean(true_slice[i])
            baselines[i] = baseline
            ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw + 2, c='#66cc66', alpha=0.9,
                    label='ground truth' if i == 0 else None)
            if ((neuron_indices[i] == 0) or (len(neuron_indices) < 50)) and stim_slice[i].mean() > 0:
                ax.plot(stim_slice[i] - baseline + i * step_v, linewidth=0.7, c='red', alpha=0.9,
                        linestyle='--', label='visual input' if i == 0 else None)

        # predictions (black, thin)
        for i in range(len(neuron_indices)):
            baseline = baselines[i]
            ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=0.7,
                    label='prediction' if i == 0 else None, c='black')

        for i in range(len(neuron_indices)):
            type_idx = selected_types[i]
            ax.text(-50, i * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize,
                    va='bottom', ha='right', color='black')

        ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices) // 50))])
        ax.set_yticks([])
        ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
        ax.set_xticklabels([start_frame, end_frame // 2, end_frame], fontsize=16)
        ax.set_xlabel('frame', fontsize=20)
        ax.set_xlim([-50, end_frame - start_frame + 100])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(loc='upper right', fontsize=14, frameon=False)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/rollout_{filename_}_{sim.visual_input_type}_{fig_name}{test_suffix}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Save activity arrays
    np.save(f"{results_dir}/activity_true{test_suffix}.npy", activity_true)
    np.save(f"{results_dir}/activity_pred{test_suffix}.npy", activity_pred)

    logger.debug(f'rollout plots saved to {results_dir}/')


def data_test_flyvis_special(
        config,
        visualize=True,
        style="color",
        verbose=False,
        best_model=None,
        step=5,
        n_rollout_frames=600,
        test_mode='',
        new_params=None,
        device=None,
        rollout_without_noise: bool = False,
        log_file=None,
):


    if "black" in style:
        plt.style.use("dark_background")
        mc = 'white'
    else:
        plt.style.use("default")
        mc = 'black'

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    log_dir = log_path(config.config_file)

    torch.random.fork_rng(devices=device)
    if sim.seed is not None:
        torch.random.manual_seed(sim.seed)
        np.random.seed(sim.seed)

    logger.info(
        f"testing... {model_config.particle_model_name} {model_config.mesh_model_name} seed: {sim.seed}")


    if tc.training_selected_neurons:
        n_neurons = 13741
        n_neuron_types = 1736
    else:
        n_neurons = sim.n_neurons
        n_neuron_types = sim.n_neuron_types

    logger.info(f"noise_model_level: {sim.noise_model_level}")
    warm_up_length = 100

    run = 0

    extent = 8
    # Import only what's needed for mixed functionality
    import flyvis
    from flyvis import Network, NetworkView
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    from flyvis_gnn.generators.flyvis_ode import (
        FlyVisODE,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )
    from flyvis_gnn.utils import setup_flyvis_model_path

    setup_flyvis_model_path()
    # Initialize datasets
    if "DAVIS" in sim.visual_input_type or "mixed" in sim.visual_input_type:
        # determine dataset roots: use config list if provided, otherwise fall back to default
        if sim.datavis_roots:
            datavis_root_list = [os.path.join(r, "JPEGImages/480p") for r in sim.datavis_roots]
        else:
            datavis_root_list = [os.path.join(get_datavis_root_dir(), "JPEGImages/480p")]

        for root in datavis_root_list:
            assert os.path.exists(root), f"video data not found at {root}"

        video_config = {
            "n_frames": 50,
            "max_frames": 80,
            "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False,
            "shuffle_sequences": True,
            "shuffle_seed": sim.seed,
        }

        # create dataset(s)
        if len(datavis_root_list) == 1:
            davis_dataset = AugmentedVideoDataset(root_dir=datavis_root_list[0], **video_config)
        else:
            datasets = [AugmentedVideoDataset(root_dir=root, **video_config) for root in datavis_root_list]
            davis_dataset = CombinedVideoDataset(datasets)
            logger.info(f"combined {len(datasets)} video datasets: {len(davis_dataset)} total sequences")
    else:
        davis_dataset = None

    if "DAVIS" in sim.visual_input_type:
        stimulus_dataset = davis_dataset
    else:
        sintel_config = {
            "sintel_path": flyvis.sintel_dir,
            "n_frames": 19,
            "flip_axes": [0, 1],
            "n_rotations": [0, 1, 2, 3, 4, 5],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 3,
            "center_crop_fraction": 0.7
        }
        stimulus_dataset = AugmentedSintel(**sintel_config)

    # Initialize network
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    ode_params = FlyVisODEParams.from_flyvis_network(net, device=device)
    edge_index = ode_params.edge_index

    if sim.n_extra_null_edges > 0:
        logger.info(f"adding {sim.n_extra_null_edges} extra null edges (mode={sim.null_edges_mode})...")
        import random
        src_np = edge_index[0].cpu().numpy()
        dst_np = edge_index[1].cpu().numpy()
        existing_edges = set(zip(src_np, dst_np))
        extra_edges = []

        if sim.null_edges_mode == 'per_column':
            from collections import Counter
            out_degree = Counter(src_np.tolist())
            total_real = edge_index.shape[1]
            ratio = sim.n_extra_null_edges / total_real
            targets_by_source = {}
            for s, d in zip(src_np, dst_np):
                targets_by_source.setdefault(int(s), set()).add(int(d))
            all_neurons = list(range(n_neurons))
            for source in range(n_neurons):
                deg = out_degree.get(source, 0)
                if deg == 0:
                    continue
                n_false = max(1, int(round(deg * ratio)))
                existing_targets = targets_by_source.get(source, set())
                candidates = [t for t in all_neurons if t != source and t not in existing_targets]
                if len(candidates) <= n_false:
                    chosen = candidates
                else:
                    chosen = random.sample(candidates, n_false)
                for t in chosen:
                    extra_edges.append([source, t])
                    existing_targets.add(t)
            logger.info(f"per_column: added {len(extra_edges)} false edges "
                        f"(requested ratio {ratio:.2f}, effective {len(extra_edges)/total_real:.2f})")
        else:
            max_attempts = sim.n_extra_null_edges * 10
            attempts = 0
            while len(extra_edges) < sim.n_extra_null_edges and attempts < max_attempts:
                source = random.randint(0, n_neurons - 1)
                target = random.randint(0, n_neurons - 1)
                if (source, target) not in existing_edges and source != target:
                    extra_edges.append([source, target])
                    existing_edges.add((source, target))
                attempts += 1

        if extra_edges:
            extra_edge_index = torch.tensor(extra_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
            ode_params.edge_index = edge_index
            ode_params.W = torch.cat([ode_params.W, torch.zeros(len(extra_edges), device=device)])

    pde = FlyVisODE(ode_params=ode_params, g_phi=torch.nn.functional.relu, params=sim.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)
    pde_modified = FlyVisODE(ode_params=ode_params.clone(), g_phi=torch.nn.functional.relu, params=sim.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)


    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)


    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        logger.info(f'best model: {best_model}')
    netname = f"{log_dir}/models/best_model_with_0_graphs_{best_model}.pt"
    logger.info(f'load {netname} ...')
    state_dict = torch.load(netname, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    model.load_state_dict(state_dict['model_state_dict'])

    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    unique_types, node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device)

    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0), device=device)]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=sim.delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    calcium_init = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=X1,
        group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
        neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
        voltage=initial_state,
        stimulus=net.stimulus().squeeze(),
        calcium=calcium_init,
        fluorescence=sim.calcium_alpha * calcium_init + sim.calcium_beta,
    )

    if tc.training_selected_neurons:
        selected_neuron_ids = tc.selected_neuron_ids
        selected_neuron_ids = np.array(selected_neuron_ids).astype(int)
        logger.info(f'testing single neuron id {selected_neuron_ids} ...')
        x_selected = x.subset(selected_neuron_ids)

    # Mixed sequence setup
    if "mixed" in sim.visual_input_type:
        mixed_types = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]  # Different lengths for each type
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        if not davis_dataset:
            sintel_config_mixed = {
                "n_frames": 19,
                "flip_axes": [0, 1],
                "n_rotations": [0, 1, 2, 3, 4, 5],
                "temporal_split": True,
                "dt": sim.delta_t,
                "interpolate": True,
                "boxfilter": dict(extent=extent, kernel_size=13),
                "vertical_splits": 3,
                "center_crop_fraction": 0.7
            }
            davis_dataset = AugmentedSintel(**sintel_config_mixed)
        sintel_iter = iter(stimulus_dataset)
        davis_iter = iter(davis_dataset)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0

    target_frames = n_rollout_frames

    if 'full' in test_mode:
        target_frames = sim.n_frames
        step = 25000
    else:
        step = 10
    logger.info(f'plot activity frames 0-{target_frames}...')

    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence
    num_passes_needed = (target_frames // total_frames_per_pass) + 1

    y_list = []
    x_list = []
    x_generated_list = []
    x_generated_modified_list = []

    x_generated = x.clone()
    x_generated_modified = x.clone()

    # Initialize RNN hidden state
    if 'RNN' in model_config.signal_model_name:
        h_state = None
    if 'LSTM' in model_config.signal_model_name:
        h_state = None
        c_state = None

    it = sim.start_frame
    id_fig = 0

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    tile_contrast = sim.tile_contrast
    n_columns = sim.n_input_neurons // 8
    tile_seed = sim.seed

    edges = ode_params.edge_index

    if ('test_ablation' in test_mode) & ('MLP' not in model_config.signal_model_name) & ('RNN' not in model_config.signal_model_name) & ('LSTM' not in model_config.signal_model_name):
        #  test_mode="test_ablation_100"
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        if ablation_ratio > 0:
            logger.info(f'test ablation ratio {ablation_ratio}')
        n_ablation = int(edges.shape[1] * ablation_ratio)
        index_ablation = np.random.choice(np.arange(edges.shape[1]), n_ablation, replace=False)

        with torch.no_grad():
            pde.ode_params.W[index_ablation] = 0
            pde_modified.ode_params.W[index_ablation] = 0
            model.W[index_ablation] = 0

    if 'test_modified' in test_mode:
        noise_W = float(test_mode.split('_')[-1])
        if noise_W > 0:
            logger.info(f'test modified W with noise level {noise_W}')
            noise_p_W = torch.randn_like(pde.ode_params.W) * noise_W
            pde_modified.ode_params.W = pde.ode_params.W.clone() + noise_p_W

        plot_weight_comparison(pde.ode_params.W, pde_modified.ode_params.W, f"{log_dir}/results/weight_comparison_{noise_W}.png")


    fig_style = dark_style
    index_to_name = INDEX_TO_NAME


    # Main loop #####################################

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data", ncols=100)):

                sequences = data["lum"]
                # Sample flash parameters for each subsequence if flash stimulus is requested
                if "flash" in sim.visual_input_type:
                    # Sample flash duration from specific values: 1, 2, 5, 10, 20 frames
                    flash_duration_options = [1, 2, 5] #, 10, 20]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]

                    flash_intensity = torch.abs(torch.rand(sim.n_input_neurons, device=device) * 0.5 + 0.5)
                if "mixed" in sim.visual_input_type:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_dataset)
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                        sequences = current_sintel_seq["lum"]
                        start_frame = sintel_frame_idx
                    elif current_type == "davis":
                        if current_davis_seq is None or davis_frame_idx >= current_davis_seq["lum"].shape[0]:
                            try:
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                            except StopIteration:
                                davis_iter = iter(davis_dataset)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0
                # Determine sequence length based on stimulus type
                if "flash" in sim.visual_input_type:
                    sequence_length = 60  # Fixed 60 frames for flash sequences
                else:
                    sequence_length = sequences.shape[0]

                for frame_id in range(sequence_length):

                    if "flash" in sim.visual_input_type:
                        # Generate repeating flash stimulus
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)  # Create on/off cycle
                        x.stimulus[:] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x.stimulus[:sim.n_input_neurons] = flash_intensity
                    elif "mixed" in sim.visual_input_type:
                        current_type = mixed_types[mixed_current_type]

                        if current_type == "blank":
                            x.stimulus[:] = 0
                        elif current_type == "noise":
                            x.stimulus[:sim.n_input_neurons] = torch.relu(
                                0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device) * 0.5)
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x.stimulus = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # 1) Cluster photoreceptors into columns based on (u,v)
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=tile_seed
                            )  # shape: (sim.n_input_neurons,)

                            # 2) Build per-column m-sequences (±1) with random phase per column
                            base = mseq_bits(p=8, seed=tile_seed).astype(np.float32)  # ±1, shape (255,)
                            rng = np.random.RandomState(tile_seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)  # (n_columns, 255), ±1

                            # 3) Convert to torch on the right device/dtype; keep as ±1 (no [0,1] mapping here)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(x.device,
                                                                                  dtype=torch.float32)  # (n_columns, 255), ±1
                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device,
                                                                              dtype=torch.long)  # (sim.n_input_neurons,)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0

                        # 4) Baseline for all neurons (mean luminance), then write per-column values to PRs
                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs
                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )
                        # Map to [0,1] with your contrast convention and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # Label columns and build neighborhood graph
                            tile_labels_np, col_centers = compute_column_labels(u_coords, v_coords, n_columns, seed=tile_seed)
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform
                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist()) for i in
                                       range(len(col_centers))]

                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            # Pre-generate ±1 codes (keep ±1; no [0,1] mapping here)
                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=x.device)
                            rng = np.random.RandomState(tile_seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)  # boolean mask
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)  # ±1
                                # NOTE: do not apply flip prob here; we do it uniformly via the helper per frame below
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(x.device, dtype=torch.float32)

                        # Baseline luminance
                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs

                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )

                        # Map to [0,1] with contrast and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    else:
                        frame = sequences[frame_id][None, None]
                        net.stimulus.add_input(frame)
                        if (sim.only_noise_visual_input > 0):
                            if (sim.visual_input_type == "") | (it == 0) | ("50/50" in sim.visual_input_type):
                                x.stimulus[:sim.n_input_neurons] = torch.relu(
                                    0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32,
                                                     device=device) * sim.only_noise_visual_input / 2)
                        else:
                            if 'blank' in sim.visual_input_type:
                                if (data_idx % sim.blank_freq > 0):
                                    x.stimulus = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[:sim.n_input_neurons] = x.stimulus[:sim.n_input_neurons] + torch.randn(sim.n_input_neurons,
                                                                                                  dtype=torch.float32,
                                                                                                  device=device) * sim.noise_visual_input

                    x_generated.stimulus = x.stimulus.clone()
                    y_generated = pde(x_generated, edge_index, has_field=False)

                    x_generated_modified.stimulus = x.stimulus.clone()
                    y_generated_modified = pde_modified(x_generated_modified, edge_index, has_field=False)

                    if 'visual' in model_config.field_type:
                        visual_input = model.forward_visual(x, it)
                        x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                        x.stimulus[model.n_input_neurons:] = 0

                    # Prediction step
                    if tc.training_selected_neurons:
                        x_selected.stimulus = x.stimulus[selected_neuron_ids].clone().detach()
                        if 'RNN' in model_config.signal_model_name:
                            y, h_state = model(x_selected.to_packed(), h=h_state, return_all=True)
                        elif 'LSTM' in model_config.signal_model_name:
                            y, h_state, c_state = model(x_selected.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif 'MLP_ODE' in model_config.signal_model_name:
                            v = x_selected.voltage.unsqueeze(-1)
                            I = x_selected.stimulus.unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif 'MLP' in model_config.signal_model_name:
                            y = model(x_selected.to_packed(), data_id=None, return_all=False)

                    else:
                        if 'RNN' in model_config.signal_model_name:
                            y, h_state = model(x.to_packed(), h=h_state, return_all=True)
                        elif 'LSTM' in model_config.signal_model_name:
                            y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif 'MLP_ODE' in model_config.signal_model_name:
                            v = x.voltage.unsqueeze(-1)
                            I = x.stimulus[:sim.n_input_neurons].unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif 'MLP' in model_config.signal_model_name:
                            y = model(x.to_packed(), data_id=None, return_all=False)
                        elif tc.neural_ODE_training:
                            data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)
                            v0 = x.voltage.flatten()
                            v_final, _ = integrate_neural_ode_FlyVis(
                                model=model,
                                v0=v0,
                                x_template=x,
                                edge_index=edge_index,
                                data_id=data_id,
                                time_steps=1,
                                delta_t=sim.delta_t,
                                neurons_per_sample=n_neurons,
                                batch_size=1,
                                has_visual_field='visual' in model_config.field_type,
                                x_ts=None,
                                device=device,
                                k_batch=torch.tensor([it], device=device),
                                ode_method=tc.ode_method,
                                rtol=tc.ode_rtol,
                                atol=tc.ode_atol,
                                adjoint=False,
                                noise_level=0.0
                            )
                            y = (v_final.view(-1, 1) - x.voltage.unsqueeze(-1)) / sim.delta_t
                        else:
                            data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)
                            y = model(x, edge_index, data_id=data_id, return_all=False)

                    # Save states (pack to legacy (N, 9) numpy for downstream analysis)
                    x_generated_list.append(to_numpy(x_generated.to_packed().clone().detach()))
                    x_generated_modified_list.append(to_numpy(x_generated_modified.to_packed().clone().detach()))

                    if tc.training_selected_neurons:
                        x_list.append(to_numpy(x_selected.to_packed().clone().detach()))
                    else:
                        x_list.append(to_numpy(x.to_packed().clone().detach()))

                    # Integration step
                    # Optionally disable process noise at test time, even if model was trained with noise
                    effective_noise_level = 0.0 if rollout_without_noise else sim.noise_model_level
                    if effective_noise_level > 0:
                        x_generated.voltage = x_generated.voltage + sim.delta_t * y_generated.squeeze(-1) + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * effective_noise_level
                        x_generated_modified.voltage = x_generated_modified.voltage + sim.delta_t * y_generated_modified.squeeze(-1) + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * effective_noise_level
                    else:
                        x_generated.voltage = x_generated.voltage + sim.delta_t * y_generated.squeeze(-1)
                        x_generated_modified.voltage = x_generated_modified.voltage + sim.delta_t * y_generated_modified.squeeze(-1)

                    if tc.training_selected_neurons:
                        if 'MLP_ODE' in model_config.signal_model_name:
                            x_selected.voltage = x_selected.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x_selected.voltage = x_selected.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and ('RNN' in model_config.signal_model_name or 'LSTM' in model_config.signal_model_name):
                            x_selected.voltage = x_generated.voltage[selected_neuron_ids].clone()
                    else:
                        if 'MLP_ODE' in model_config.signal_model_name:
                            x.voltage = x.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and ('RNN' in model_config.signal_model_name):
                            x.voltage = x_generated.voltage.clone()

                    # Guard against NaN / divergence from a poorly trained model
                    v_model = x_selected.voltage if tc.training_selected_neurons else x.voltage
                    if torch.isnan(v_model).any() or torch.isinf(v_model).any():
                        logger.error(f"rollout diverged at iteration {it} (NaN/Inf in voltage) — aborting")
                        break
                    if tc.training_selected_neurons:
                        x_selected.voltage = torch.clamp(x_selected.voltage, min=-100.0, max=100.0)
                    else:
                        x.voltage = torch.clamp(x.voltage, min=-100.0, max=100.0)

                    if sim.calcium_type == "leaky":
                        # Voltage-driven activation
                        if sim.calcium_activation == "softplus":
                            u = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            u = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            u = torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            u = x.voltage.clone()

                        x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + u)
                        x.calcium = torch.clamp(x.calcium, min=0.0)
                        x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta

                        y = (x.calcium - torch.tensor(x_list[-1][:, 7], dtype=torch.float32, device=device)).unsqueeze(-1) / sim.delta_t

                    y_list.append(to_numpy(y.clone().detach()))

                    if (it > 0) & (it < 100) & (it % step == 0) & visualize & (not tc.training_selected_neurons):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy(x.pos),
                            voltages=to_numpy(x.voltage),
                            stimulus=to_numpy(x.stimulus[:sim.n_input_neurons]),
                            neuron_types=to_numpy(x.neuron_type).astype(int),
                            output_path=f"{log_dir}/tmp_recons/Fig_{run}_{num}.png",
                            calcium=to_numpy(x.calcium) if sim.calcium_type != "none" else None,
                            n_input_neurons=sim.n_input_neurons,
                            style=fig_style,
                        )

                    it = it + 1
                    if it >= target_frames:
                        break
                if it >= target_frames:
                    break

            if it >= target_frames:
                break
    logger.info(f"generated {len(x_list)} frames total")


    if visualize:
        logger.info('generating lossless video ...')

        output_name = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'
        src = f"{log_dir}/tmp_recons/Fig_0_000000.png"
        dst = f"{log_dir}/results/input_{output_name}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=f"{log_dir}/results", run=run,
                                        output_name=output_name,framerate=20)

        # files = glob.glob(f'./{log_dir}/tmp_recons/*')
        # for f in files:
        #     os.remove(f)


    x_list = np.array(x_list)
    x_generated_list = np.array(x_generated_list)
    x_generated_modified_list = np.array(x_generated_modified_list)
    y_list = np.array(y_list)

    neuron_types = node_types_int

    if sim.calcium_type != "none":
        # Use calcium (index 7)
        activity_true = x_generated_list[:, :, 7].squeeze().T  # (n_neurons, n_frames)
        activity_pred = x_list[:, :, 7].squeeze().T
    else:
        # Use voltage (index 3)
        activity_true = x_generated_list[:, :, 3].squeeze().T
        visual_input_true = x_generated_list[:, :, 4].squeeze().T
        activity_true_modified = x_generated_modified_list[:, :, 3].squeeze().T
        activity_pred = x_list[:, :, 3].squeeze().T


    start_frame = 0
    end_frame = target_frames


    if tc.training_selected_neurons:           # MLP, RNN and ODE are trained on limted number of neurons

        logger.info(f"evaluating on selected neurons only: {selected_neuron_ids}")
        x_generated_list = x_generated_list[:, selected_neuron_ids, :]
        x_generated_modified_list = x_generated_modified_list[:, selected_neuron_ids, :]
        neuron_types = neuron_types[selected_neuron_ids]

        true_slice = activity_true[selected_neuron_ids, start_frame:end_frame]
        visual_input_slice = visual_input_true[selected_neuron_ids, start_frame:end_frame]
        pred_slice = activity_pred[start_frame:end_frame]

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(true_slice, pred_slice, "selected neurons")

        # Log rollout metrics to file
        rollout_log_path = f"{log_dir}/results_rollout.log"
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for Selected Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {np.nanmean(pearson_all):.3f} ± {np.nanstd(pearson_all):.3f} [{np.nanmin(pearson_all):.3f}, {np.nanmax(pearson_all):.3f}]\n")
            # f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            # f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(selected_neuron_ids)}\n")

        if len(selected_neuron_ids)==1:
            pred_slice = pred_slice[None,:]

        filename_ = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'

        # Determine which figures to create
        if len(selected_neuron_ids) > 50:
            # Create sample: take the last 10 neurons from selected_neuron_ids
            sample_indices = list(range(len(selected_neuron_ids) - 10, len(selected_neuron_ids)))

            figure_configs = [
                ("all", list(range(len(selected_neuron_ids)))),
                ("sample", sample_indices)
            ]
        else:
            figure_configs = [("", list(range(len(selected_neuron_ids))))]

        for fig_suffix, neuron_plot_indices in figure_configs:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            step_v = 2.5
            lw = 6

            # Adjust fontsize based on number of neurons being plotted
            name_fontsize = 10 if len(neuron_plot_indices) > 50 else 18

            # Plot ground truth (green, thick) — all traces first
            baselines = {}
            for plot_idx, i in enumerate(trange(len(neuron_plot_indices), ncols=100, desc=f"plotting {fig_suffix}")):
                neuron_idx = neuron_plot_indices[i]
                baseline = np.mean(true_slice[neuron_idx])
                baselines[plot_idx] = baseline
                ax.plot(true_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=lw+2, c='#66cc66', alpha=0.9,
                        label='ground truth' if plot_idx == 0 else None)
                # Plot visual input only for neuron_id = 0
                if ((selected_neuron_ids[neuron_idx] == 0) | (len(neuron_plot_indices) < 50)) and visual_input_slice[neuron_idx].mean() > 0:
                    ax.plot(visual_input_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c='yellow', alpha=0.9,
                            linestyle='--', label='visual input')

            # Plot predictions (black, thin) — on top
            for plot_idx, i in enumerate(range(len(neuron_plot_indices))):
                neuron_idx = neuron_plot_indices[i]
                baseline = baselines[plot_idx]
                ax.plot(pred_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c=mc,
                        label='prediction' if plot_idx == 0 else None)

            for plot_idx, i in enumerate(neuron_plot_indices):
                type_idx = int(to_numpy(x.neuron_type[selected_neuron_ids[i]]).item())
                ax.text(-50, plot_idx * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_plot_indices) * (step_v + 0.25 + 0.15 * (len(neuron_plot_indices)//50))])
            ax.set_yticks([])
            ax.set_xlabel('time (frames)', fontsize=20)
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14, frameon=False)
            ax.set_xlim([0, end_frame - start_frame + 100])

            plt.tight_layout()
            save_suffix = f"_{fig_suffix}" if fig_suffix else ""
            plt.savefig(f"{log_dir}/results/rollout_{filename_}_{sim.visual_input_type}{save_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

    else:

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(activity_true, activity_pred, "all neurons")

        # Log rollout metrics to file
        rollout_log_path = f"{log_dir}/results_rollout.log"
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for All Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {np.nanmean(pearson_all):.3f} ± {np.nanstd(pearson_all):.3f} [{np.nanmin(pearson_all):.3f}, {np.nanmax(pearson_all):.3f}]\n")
            # f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            # f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(activity_true)}\n")
            f.write(f"Frames evaluated: {start_frame} to {end_frame}\n")

        # Write to analysis log file for Claude
        if log_file:
            # log_file.write(f"test_R2: {np.nanmean(r2_all):.4f}\n")
            log_file.write(f"test_pearson: {np.nanmean(pearson_all):.4f}\n")

        filename_ = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'

        # Create two figures with different neuron type selections
        for fig_name, selected_types in [
            ("selected", [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]),  # L1, Mi12, Mi2, R1, T1, T4a, T5a, Tm1, Tm4, Tm9
            ("all", np.arange(0, n_neuron_types))
        ]:
            neuron_indices = []
            for stype in selected_types:
                indices = np.where(neuron_types == stype)[0]
                if len(indices) > 0:
                    neuron_indices.append(indices[0])

            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            true_slice = activity_true[neuron_indices, start_frame:end_frame]
            visual_input_slice = visual_input_true[neuron_indices, start_frame:end_frame]
            pred_slice = activity_pred[neuron_indices, start_frame:end_frame]
            step_v = 2.5
            lw = 2

            # Adjust fontsize based on number of neurons
            name_fontsize = 10 if len(selected_types) > 50 else 18

            # Plot ground truth (green, thick) — all traces first
            baselines = {}
            for i in range(len(neuron_indices)):
                baseline = np.mean(true_slice[i])
                baselines[i] = baseline
                ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw+2, c='#66cc66', alpha=0.9,
                        label='ground truth' if i == 0 else None)
                # Plot visual input for neuron 0 OR when fewer than 50 neurons
                if ((neuron_indices[i] == 0) | (len(neuron_indices) < 50)) and visual_input_slice[i].mean() > 0:
                    ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=0.7, c='red', alpha=0.9,
                            linestyle='--', label='visual input')

            # Plot predictions (black, thin) — on top
            for i in range(len(neuron_indices)):
                baseline = baselines[i]
                ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=0.7, label='prediction' if i == 0 else None, c=mc)


            for i in range(len(neuron_indices)):
                type_idx = selected_types[i]
                ax.text(-50, i * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices)//50))])
            ax.set_yticks([])
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)
            ax.set_xlabel('time (frames)', fontsize=20)
            ax.set_xlim([-50, end_frame - start_frame + 100])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14, frameon=False)

            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/rollout_{filename_}_{sim.visual_input_type}_{fig_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        if ('test_ablation' in test_mode) or ('test_inactivity' in test_mode):
            np.save(f"{log_dir}/results/activity_modified.npy", activity_true_modified)
            np.save(f"{log_dir}/results/activity_modified_pred.npy", activity_pred)
        else:
            np.save(f"{log_dir}/results/activity_true.npy", activity_true)
            np.save(f"{log_dir}/results/activity_pred.npy", activity_pred)


