import logging
import os
import time
import warnings

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from flyvis_gnn.figure_style import default_style
from flyvis_gnn.log import get_logger
from flyvis_gnn.metrics import compute_dynamics_r2
from flyvis_gnn.models.Neural_ode_wrapper_FlyVis import (
    debug_check_gradients,
    neural_ode_loss_FlyVis,
)
from flyvis_gnn.models.recurrent_step import recurrent_loss
from flyvis_gnn.models.registry import create_model
from flyvis_gnn.models.training_utils import build_lr_scheduler, build_model, determine_load_fields, load_flyvis_data
from flyvis_gnn.models.utils import (
    LossRegularizer,
    _batch_frames,
    analyze_data_svd,
    set_trainable_parameters,
)
from flyvis_gnn.plot import (
    plot_signal_loss,
    plot_training_flyvis,
    plot_training_linear,
    plot_training_summary_panels,
    render_visual_field_video,
)
from flyvis_gnn.sparsify import clustering_evaluation, umap_cluster_reassign
from flyvis_gnn.utils import (
    CustomColorMap,
    check_and_clear_memory,
    create_log_dir,
    graphs_data_path,
    to_numpy,
)

_logger = get_logger(__name__)

ANSI_RESET = '\033[0m'
ANSI_GREEN = '\033[92m'
ANSI_YELLOW = '\033[93m'
ANSI_ORANGE = '\033[38;5;208m'
ANSI_RED = '\033[91m'

def r2_color(val, thresholds=(0.9, 0.7, 0.3)):
    """ANSI color for an R² value: green > t0, yellow > t1, orange > t2, red otherwise."""
    t0, t1, t2 = thresholds
    return ANSI_GREEN if val > t0 else ANSI_YELLOW if val > t1 else ANSI_ORANGE if val > t2 else ANSI_RED


def data_train(config=None, erase=False, best_model=None, style=None, device=None, log_file=None):
    # plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    _logger.info(f"dataset: {config.dataset}")
    _logger.info(f"{config.description}")

    if 'fly' in config.dataset:
        if 'RNN' in config.graph_model.signal_model_name or 'LSTM' in config.graph_model.signal_model_name:
            data_train_flyvis_RNN(config, erase, best_model, device)
        else:
            data_train_flyvis(config, erase, best_model, device, log_file=log_file)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")

    _logger.info("training completed.")


def data_train_flyvis(config, erase, best_model, device, log_file=None):
    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    replace_with_cluster = 'replace' in tc.sparsity
    umap_cluster_active = tc.umap_cluster_method != 'none'

    if config.training.seed != 42:
        torch.random.fork_rng(devices=device)
        torch.random.manual_seed(config.training.seed)

    default_style.apply_globally()

    if 'visual' in model_config.field_type:
        has_visual_field = True
        if 'instantNGP' in model_config.field_type:
            _logger.info('train with visual field instantNGP')
        else:
            _logger.info('train with visual field NNR')
    else:
        has_visual_field = False
    if 'test' in model_config.field_type:
        test_neural_field = True
        _logger.info('train with test field NNR')
    else:
        test_neural_field = False

    log_dir, logger = create_log_dir(config, erase)

    load_fields = determine_load_fields(config)
    x_ts, y_ts, type_list = load_flyvis_data(
        config.dataset, split='train', fields=load_fields, device=device,
        training_selected_neurons=tc.training_selected_neurons,
        selected_neuron_ids=tc.selected_neuron_ids if tc.training_selected_neurons else None,
        measurement_noise_level=sim.measurement_noise_level,
    )

    # get n_neurons and n_frames from data, not config file
    n_neurons = x_ts.n_neurons
    config.simulation.n_neurons = n_neurons
    sim.n_frames = x_ts.n_frames
    _logger.info(f'dataset: {x_ts.n_frames} frames,  n neurons: {n_neurons}')
    logger.info(f'n neurons: {n_neurons}')

    xnorm = x_ts.xnorm
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    _logger.info(f'xnorm: {to_numpy(xnorm):0.3f}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    _logger.info(f'ynorm: {to_numpy(ynorm):0.3f}')
    logger.info(f'ynorm: {to_numpy(ynorm)}')

    # SVD analysis of activity and visual stimuli (skip if already exists)
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    if not os.path.exists(svd_plot_path):
        analyze_data_svd(x_ts, log_dir, config=config, logger=logger, is_flyvis=True)
    else:
        _logger.info(f'svd analysis already exists: {svd_plot_path}')

    # Load edges early so n_edges is correct before model creation
    from flyvis_gnn.generators.ode_params import FlyVisODEParams
    ode_params = FlyVisODEParams.load(graphs_data_path(config.dataset), device=device)
    gt_weights = ode_params.W
    edges = ode_params.edge_index
    actual_n_edges = edges.shape[1]
    expected_total = sim.n_edges + sim.n_extra_null_edges
    if actual_n_edges == expected_total and sim.n_extra_null_edges > 0:
        # Null edges already baked into saved data — keep n_edges and n_extra_null_edges as-is
        # so model sizes W = n_edges + n_extra_null_edges = actual_n_edges
        _logger.info(f'null edges in data: {sim.n_edges} base + {sim.n_extra_null_edges} null = {actual_n_edges}')
    elif actual_n_edges != sim.n_edges:
        # Edge removal case: actual < config, override n_edges
        _logger.info(f'n_edges mismatch: config={sim.n_edges}, actual={actual_n_edges} — using actual')
        config.simulation.n_edges = actual_n_edges
    _logger.info(f'{actual_n_edges} edges')

    # Resolve checkpoint path from best_model argument
    checkpoint_path = None
    if best_model and best_model != '' and best_model != 'None':
        checkpoint_path = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt"
    elif tc.pretrained_model != '':
        checkpoint_path = tc.pretrained_model

    reset_epoch = (tc.pretrained_model != '' and not best_model)
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path, reset_epoch=reset_epoch)
    list_loss = []

    # W init mode info
    w_init_mode = getattr(tc, 'w_init_mode', 'randn')
    if w_init_mode != 'randn':
        w_init_scale = getattr(tc, 'w_init_scale', 1.0)
        _logger.info(f'W init mode: {w_init_mode}' + (f' (scale={w_init_scale})' if w_init_mode == 'randn_scaled' else ''))

    # === LLM-MODIFIABLE: OPTIMIZER SETUP START ===
    # Change optimizer type, learning rate schedule, parameter groups

    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_total_params:,}')
    lr = tc.lr
    if tc.learning_rate_update_start == 0:
        lr_update = tc.lr
    else:
        lr_update = tc.learning_rate_update_start
    lr_embedding = tc.lr_embedding
    lr_W = tc.lr_W
    learning_rate_NNR = tc.learning_rate_NNR
    learning_rate_NNR_f = tc.learning_rate_NNR_f

    _logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_embedding {lr_embedding}, learning_rate_NNR_f {learning_rate_NNR_f}')

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, learning_rate_NNR=learning_rate_NNR, learning_rate_NNR_f = learning_rate_NNR_f)

    lr_scheduler = build_lr_scheduler(optimizer, config)
    scheduler_type = getattr(tc, 'lr_scheduler', 'none')
    if scheduler_type != 'none':
        _logger.info(f'LR scheduler: {scheduler_type}')
    # === LLM-MODIFIABLE: OPTIMIZER SETUP END ===
    model.train()

    net = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs.pt"
    _logger.info(f'network: {net}')
    _logger.info(f'initial tc.batch_size: {tc.batch_size}')

    ids = np.arange(n_neurons)

    if tc.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            # get source neurons that connect to neuron i
            mask = edges[1] == i
            index_weight.append(edges[0][mask])

    logger.info(f'coeff_W_L1: {tc.coeff_W_L1} coeff_g_phi_diff: {tc.coeff_g_phi_diff} coeff_f_theta_diff: {tc.coeff_f_theta_diff}')
    _logger.info(f'coeff_W_L1: {tc.coeff_W_L1} coeff_g_phi_diff: {tc.coeff_g_phi_diff} coeff_f_theta_diff: {tc.coeff_f_theta_diff}')
     # proximal L1 info
    coeff_proximal = getattr(tc, 'coeff_W_L1_proximal', 0.0)
    if coeff_proximal > 0:
        _logger.info(f'proximal L1 soft-thresholding on W: coeff={coeff_proximal}')

    _logger.info("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []

    regularizer = LossRegularizer(
        train_config=tc,
        model_config=model_config,
        activity_column=3,  # flyvis uses column 3 for activity
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='flyvis',
        dataset=config.dataset,
    )
    regularizer.set_activity_stats(x_ts, device)

    loss_components = {'loss': []}

    time.sleep(0.2)

    training_start_time = time.time()

    # Metrics log: tracks R2 evolution over training iterations
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    with open(metrics_log_path, 'w') as f:
        f.write('iteration,connectivity_r2,vrest_r2,tau_r2\n')

    # Valid frame range for sampling (matches np.random.randint logic it replaces)
    _frame_min_k = tc.time_window
    _frame_max_k = sim.n_frames - 4 - tc.time_step  # exclusive upper bound
    _frame_range = max(_frame_max_k - _frame_min_k, 1)

    embedding_frozen = False
    unfreeze_at_iteration = -1

    for epoch in range(start_epoch, tc.n_epochs):

        Niter = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
        plot_frequency = int(Niter // 20)
        connectivity_plot_frequency = int(Niter // 10)
        # Early-phase R2: 4 extra checkpoints in [1, connectivity_plot_frequency)
        early_r2_frequency = connectivity_plot_frequency // 5
        n_plots_per_epoch = 4
        plot_iterations = set(int(x) for x in np.linspace(Niter // n_plots_per_epoch, Niter - 1, n_plots_per_epoch))
        print(f'every {connectivity_plot_frequency} iterations: {Niter} iterations per epoch, plot '
              f'(early-phase every {early_r2_frequency} iterations)')

        # Compute unfreeze point for this epoch if embedding was frozen by UMAP clustering
        if embedding_frozen and tc.umap_cluster_fix_embedding_ratio > 0:
            unfreeze_at_iteration = int(Niter * tc.umap_cluster_fix_embedding_ratio)
        else:
            unfreeze_at_iteration = -1

        total_loss = 0
        total_loss_regul = 0
        k = 0

        loss_noise_level = tc.loss_noise_level * (0.95 ** epoch)
        regularizer.set_epoch(epoch, plot_frequency, Niter=Niter)

        # Two-phase training: epoch 0 = full LRs, epoch 1+ = reduce W/MLP, keep SIREN
        if tc.alternate_training and epoch >= 1:
            phase_mult = tc.alternate_lr_ratio
            optimizer, n_total_params = set_trainable_parameters(
                model=model,
                lr_embedding=lr_embedding * phase_mult,
                lr=lr * phase_mult,
                lr_update=lr_update * phase_mult,
                lr_W=lr_W * phase_mult,
                learning_rate_NNR=learning_rate_NNR,
                learning_rate_NNR_f=learning_rate_NNR_f,
            )
            lr_scheduler = build_lr_scheduler(optimizer, config)
            _logger.info(f'Phase 1 (SIREN focus): W/MLP LRs *= {phase_mult}, NNR_f LR = {learning_rate_NNR_f}')

        # Reproducible per-epoch frame sampling (replaces bare np.random.randint)
        epoch_rng = np.random.RandomState(tc.seed + epoch)
        frame_indices = epoch_rng.randint(0, _frame_range, size=Niter * tc.batch_size) + _frame_min_k

        last_connectivity_r2 = None
        last_vrest_r2 = 0.0
        last_tau_r2 = 0.0
        field_R2 = None
        field_slope = None
        pbar = trange(Niter, ncols=150)
        
        # === LLM-MODIFIABLE: TRAINING LOOP START ===
        # Main training loop. Suggested changes: loss function, gradient clipping,
        # data sampling strategy, LR scheduler steps, early stopping.
        # Do NOT change: function signature, model construction, data loading, return values.
        for N in pbar:

            # Unfreeze embedding at the midpoint after UMAP clustering froze it
            if embedding_frozen and N == unfreeze_at_iteration:
                embedding_frozen = False
                lr_embedding = tc.lr_embedding
                optimizer, n_total_params = set_trainable_parameters(
                    model=model, lr_embedding=lr_embedding, lr=lr,
                    lr_update=lr_update, lr_W=lr_W,
                    learning_rate_NNR=learning_rate_NNR)
                _logger.debug(f'unfreezing embedding at iteration {N}/{Niter}')

            optimizer.zero_grad()

            # Recurrent training (standard or multi-start) — delegated to recurrent_step
            if tc.recurrent_training and not tc.neural_ODE_training:
                loss, regul_val = recurrent_loss(
                    model=model, x_ts=x_ts, y_ts=y_ts, edges=edges, ids=ids,
                    frame_indices=frame_indices, iter_idx=N, config=config,
                    device=device, xnorm=xnorm, ynorm=ynorm,
                    regularizer=regularizer, has_visual_field=has_visual_field,
                )
                loss.backward()
                if hasattr(tc, 'grad_clip_W') and tc.grad_clip_W > 0 and hasattr(model, 'W'):
                    if model.W.grad is not None:
                        torch.nn.utils.clip_grad_norm_([model.W], max_norm=tc.grad_clip_W)
                optimizer.step()
                lr_scheduler.step()
                total_loss += loss.item()
                total_loss_regul += regul_val
                regularizer.finalize_iteration()

                if regularizer.should_record():
                    loss_components['loss'].append((loss.item() - regul_val) / n_neurons)
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=Niter,
                                    epoch_boundaries=regularizer.epoch_boundaries)

                # R2 checkpoint
                is_regular_r2 = (N > 0) and (N % connectivity_plot_frequency == 0)
                is_early_r2 = (N > 0) and (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
                model_name = model_config.signal_model_name
                if (is_regular_r2 or is_early_r2) and 'MLP' not in model_name:
                    last_connectivity_r2 = plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=sim.n_neuron_types)
                    last_vrest_r2, last_tau_r2 = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f}\n')

                if last_connectivity_r2 is not None:
                    c_conn, c_vr, c_tau = r2_color(last_connectivity_r2), r2_color(last_vrest_r2), r2_color(last_tau_r2)
                    pbar.set_postfix_str(f'{c_conn}conn={last_connectivity_r2:.3f}{ANSI_RESET} {c_vr}Vr={last_vrest_r2:.3f}{ANSI_RESET} {c_tau}τ={last_tau_r2:.3f}{ANSI_RESET}')
                continue

            state_batch = []
            y_list = []
            ids_list = []
            k_list = []
            visual_input_list = []
            ids_index = 0

            loss = torch.zeros(1, device=device)
            regularizer.reset_iteration()

            # Consecutive batch: pick one random start, use batch_size consecutive frames
            if tc.consecutive_batch:
                k_start = int(frame_indices[N * tc.batch_size])

            for batch in range(tc.batch_size):

                if tc.consecutive_batch:
                    k = k_start + batch
                else:
                    k = int(frame_indices[N * tc.batch_size + batch])

                if tc.recurrent_training or tc.neural_ODE_training:
                    k = k - k % tc.time_step

                x = x_ts.frame(k)

                # Add measurement noise to observed voltage
                if x.noise is not None and sim.measurement_noise_level > 0:
                    x.voltage = x.voltage + x.noise

                if tc.time_window > 0:
                    x_temporal = x_ts.voltage[k - tc.time_window + 1: k + 1].T
                    # x stays as NeuronState; x_temporal passed separately to temporal model

                if has_visual_field:
                    visual_input = model.forward_visual(x, k)
                    x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                    x.stimulus[model.n_input_neurons:] = 0

                if not (torch.isnan(x.voltage).any()):

                    if batch==0:  # apply regularization only once
                        regul_loss = regularizer.compute(
                            model=model,
                            x=x,
                            in_features=None,
                            ids=ids,
                            ids_batch=None,
                            edges=edges,
                            device=device,
                            xnorm=xnorm
                        )
                        loss = loss + regul_loss

                    if tc.recurrent_training or tc.neural_ODE_training:
                        y = x_ts.voltage[k + tc.time_step].unsqueeze(-1)
                    elif test_neural_field:
                        y = x_ts.stimulus[k, :sim.n_input_neurons].unsqueeze(-1)
                    else:
                        y = torch.tensor(y_ts[k], device=device) / ynorm     # loss on activity derivative


                    if loss_noise_level>0:
                        y = y + torch.randn(y.shape, device=device) * loss_noise_level

                    if not (torch.isnan(y).any()):

                        state_batch.append(x)
                        n = x.n_neurons
                        y_list.append(y)
                        ids_list.append(ids + ids_index)
                        k_list.append(torch.ones((n, 1), dtype=torch.int, device=device) * k)
                        if test_neural_field:
                            visual_input_list.append(visual_input)
                        ids_index += n


            if state_batch:

                data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)
                y_batch = torch.cat(y_list, dim=0)
                ids_batch = np.concatenate(ids_list, axis=0)
                k_batch = torch.cat(k_list, dim=0)

                total_loss_regul += loss.item()

                if test_neural_field:
                    visual_input_batch = torch.cat(visual_input_list, dim=0)
                    loss = loss + (visual_input_batch - y_batch).norm(2)


                elif 'MLP_ODE' in model_config.signal_model_name:
                    batched_state, _ = _batch_frames(state_batch, edges)
                    batched_x = batched_state.to_packed()
                    pred = model(batched_x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                elif 'MLP' in model_config.signal_model_name:
                    batched_state, _ = _batch_frames(state_batch, edges)
                    batched_x = batched_state.to_packed()
                    pred = model(batched_x, data_id=data_id, return_all=False)

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

                else: # 'GNN' branch

                    batched_state, batched_edges = _batch_frames(state_batch, edges)
                    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
                    loss = loss + update_regul


                    if tc.neural_ODE_training:

                        ode_state_clamp = getattr(tc, 'ode_state_clamp', 10.0)
                        ode_stab_lambda = getattr(tc, 'ode_stab_lambda', 0.0)
                        ode_loss, pred_x = neural_ode_loss_FlyVis(
                            model=model,
                            dataset_batch=state_batch,
                            edge_index=edges,
                            x_ts=x_ts,
                            k_batch=k_batch,
                            time_step=tc.time_step,
                            batch_size=tc.batch_size,
                            n_neurons=n_neurons,
                            ids_batch=ids_batch,
                            delta_t=sim.delta_t,
                            device=device,
                            data_id=data_id,
                            has_visual_field=has_visual_field,
                            y_batch=y_batch,
                            noise_level=tc.noise_recurrent_level,
                            ode_method=tc.ode_method,
                            rtol=tc.ode_rtol,
                            atol=tc.ode_atol,
                            adjoint=tc.ode_adjoint,
                            iteration=N,
                            state_clamp=ode_state_clamp,
                            stab_lambda=ode_stab_lambda
                        )
                        loss = loss + ode_loss


                    elif tc.recurrent_training:

                        pred_x = batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                        if tc.time_step > 1:
                            for step in range(tc.time_step - 1):
                                neurons_per_sample = state_batch[0].n_neurons

                                for b in range(tc.batch_size):
                                    start_idx = b * neurons_per_sample
                                    end_idx = (b + 1) * neurons_per_sample

                                    state_batch[b].voltage = pred_x[start_idx:end_idx].squeeze()

                                    k_current = k_batch[start_idx, 0].item() + step + 1

                                    if has_visual_field:
                                        visual_input_next = model.forward_visual(state_batch[b], k_current)
                                        state_batch[b].stimulus[:model.n_input_neurons] = visual_input_next.squeeze(-1)
                                        state_batch[b].stimulus[model.n_input_neurons:] = 0
                                    else:
                                        x_next = x_ts.frame(k_current)
                                        state_batch[b].stimulus = x_next.stimulus

                                batched_state, batched_edges = _batch_frames(state_batch, edges)
                                pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                                pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                        loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * tc.time_step)).norm(2)


                    else:

                        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)


                # === LLM-MODIFIABLE: BACKWARD AND STEP START ===
                # Allowed changes: gradient clipping, LR scheduler step, loss scaling
                loss.backward()

                # debug gradient check for neural ODE training
                if tc.neural_ODE_training and (N % 500 == 0):
                    debug_check_gradients(model, loss, N)

                # W-specific gradient clipping: clip W gradients to force optimizer
                # to adjust lin_update (which contains V_rest/tau) instead of W
                if hasattr(tc, 'grad_clip_W') and tc.grad_clip_W > 0 and hasattr(model, 'W'):
                    if model.W.grad is not None:
                        torch.nn.utils.clip_grad_norm_([model.W], max_norm=tc.grad_clip_W)

                optimizer.step()
                lr_scheduler.step()
                # === LLM-MODIFIABLE: BACKWARD AND STEP END ===

                total_loss += loss.item()
                total_loss_regul += regularizer.get_iteration_total()

                # finalize iteration to record history
                regularizer.finalize_iteration()


                if regularizer.should_record():
                    # get history from regularizer and add loss component
                    current_loss = loss.item()
                    regul_total_this_iter = regularizer.get_iteration_total()
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}

                    # pass per-neuron normalized values to debug (to match dictionary values)
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=Niter,
                                   epoch_boundaries=regularizer.epoch_boundaries, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)

                    # persist full loss decomposition so the plot can be regenerated
                    torch.save({
                        **plot_dict,
                        'epoch_boundaries': list(regularizer.epoch_boundaries),
                    }, os.path.join(log_dir, 'loss_components.pt'))

                    if tc.save_all_checkpoints:
                        torch.save(
                            {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

                # R2 checkpoint: regular interval + early-phase extra points
                is_regular_r2 = (N > 0) and (N % connectivity_plot_frequency == 0)
                is_early_r2 = (N > 0) and (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
                model_name = model_config.signal_model_name
                if (is_regular_r2 or is_early_r2) and not test_neural_field and 'linear' in model_name:
                    last_connectivity_r2, last_tau_r2, last_vrest_r2 = plot_training_linear(
                        model, config, epoch, N, log_dir, device, gt_weights, n_neurons=n_neurons)
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f}\n')
                elif (is_regular_r2 or is_early_r2) and not test_neural_field and 'MLP' not in model_name:
                    last_connectivity_r2 = plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=sim.n_neuron_types)
                    last_vrest_r2, last_tau_r2 = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f}\n')

                if last_connectivity_r2 is not None:
                    c_conn, c_vr, c_tau = r2_color(last_connectivity_r2), r2_color(last_vrest_r2), r2_color(last_tau_r2)
                    pbar.set_postfix_str(f'{c_conn}conn={last_connectivity_r2:.3f}{ANSI_RESET} {c_vr}Vr={last_vrest_r2:.3f}{ANSI_RESET} {c_tau}τ={last_tau_r2:.3f}{ANSI_RESET}')


                if (has_visual_field) & (N in plot_iterations):
                    field_R2, field_slope = render_visual_field_video(
                        model, x_ts, sim, log_dir, epoch, N, logger)


                    if last_connectivity_r2 is not None:
                        pbar.set_postfix_str(f'{r2_color(last_connectivity_r2)}R²={last_connectivity_r2:.3f}{ANSI_RESET}')
                    if tc.save_all_checkpoints:
                        torch.save(
                            {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        # === LLM-MODIFIABLE: TRAINING LOOP END ===

        # Calculate epoch-level losses
        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        _logger.info("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'))

        if has_visual_field and hasattr(model, 'NNR_f'):
            torch.save(model.NNR_f.state_dict(),
                       os.path.join(log_dir, 'models', f'inr_stimulus_{epoch}.pt'))

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(3 * default_style.figure_height * default_style.default_aspect,
                                    2 * default_style.figure_height))

        # Plot 1: Loss
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(list_loss, color=default_style.foreground, linewidth=default_style.line_width)
        ax1.set_xlim([0, tc.n_epochs])
        default_style.ylabel(ax1, 'loss')
        default_style.xlabel(ax1, 'epochs')

        plot_training_summary_panels(fig, log_dir, Niter=Niter)

        if replace_with_cluster:

            if (epoch % tc.sparsity_freq == tc.sparsity_freq - 1) & (epoch < tc.n_epochs - tc.sparsity_freq):
                _logger.info('replace embedding with clusters ...')
                eps = tc.cluster_distance_threshold
                results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                _logger.info(f"eps={eps}: {results['n_clusters_found']} clusters, "
                      f"accuracy={results['accuracy']:.3f}")

                labels = results['cluster_labels']

                for n in np.unique(labels):
                    # if n == -1:
                    #     continue
                    indices = np.where(labels == n)[0]
                    if len(indices) > 1:
                        with torch.no_grad():
                            model.a[indices, :] = torch.mean(model.a[indices, :], dim=0, keepdim=True)

                fig.add_subplot(2, 3, 6)
                type_cmap = CustomColorMap(config=config)
                for n in range(sim.n_neuron_types):
                    pos = torch.argwhere(type_list == n)
                    plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=5, color=type_cmap.color(n),
                                alpha=0.7, edgecolors='none')
                plt.xlabel('embedding 0', fontsize=18)
                plt.ylabel('embedding 1', fontsize=18)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.5, 0.9, f"eps={eps}: {results['n_clusters_found']} clusters, accuracy={results['accuracy']:.3f}")

                if tc.fix_cluster_embedding:
                    lr_embedding = 1.0E-10
                    # the embedding is fixed for 1 epoch

            else:
                lr = tc.lr
                lr_embedding = tc.lr_embedding
                lr_W = tc.lr_W
                learning_rate_NNR = tc.learning_rate_NNR

            logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}, learning_rate_NNR {learning_rate_NNR}')
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W,
                                                                 learning_rate_NNR=learning_rate_NNR)

        if umap_cluster_active:
            if (epoch % tc.umap_cluster_freq == tc.umap_cluster_freq - 1) & (epoch < tc.n_epochs - 1):
                _logger.info('UMAP cluster reassign ...')
                umap_results = umap_cluster_reassign(
                    model, config, x_ts, edges, n_neurons, type_list, device, logger=logger,
                    reinit_mlps=tc.umap_cluster_reinit_mlps,
                    relearn_epochs=tc.umap_cluster_relearn_epochs)

                if umap_results is not None:
                    fig.add_subplot(2, 3, 6)
                    type_cmap = CustomColorMap(config=config)
                    a_umap = umap_results['a_umap']
                    for n_type in range(sim.n_neuron_types):
                        pos = torch.argwhere(type_list == n_type)
                        pos_np = to_numpy(pos).flatten()
                        plt.scatter(a_umap[pos_np, 0], a_umap[pos_np, 1], s=5,
                                    color=type_cmap.color(n_type), alpha=0.7, edgecolors='none')
                    plt.xlabel(r'UMAP$_1$', fontsize=12)
                    plt.ylabel(r'UMAP$_2$', fontsize=12)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(f"{umap_results['n_clusters']} cl, acc={umap_results['accuracy']:.3f}", fontsize=10)

                if tc.umap_cluster_fix_embedding or tc.umap_cluster_fix_embedding_ratio > 0:
                    lr_embedding = 1.0E-10
                    embedding_frozen = True

                # rebuild optimizer to reset momentum and relearn f_theta/g_phi
                optimizer, n_total_params = set_trainable_parameters(
                    model=model, lr_embedding=lr_embedding, lr=lr,
                    lr_update=lr_update, lr_W=lr_W,
                    learning_rate_NNR=learning_rate_NNR)

        plt.tight_layout()
        plt.savefig(f"{log_dir}/tmp_training/epoch_{epoch}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # Calculate and log training time
    training_time = time.time() - training_start_time
    training_time_min = training_time / 60.0
    _logger.info(f"training completed in {training_time_min:.1f} minutes")
    logger.info(f"training completed in {training_time_min:.1f} minutes")

    if log_file is not None:
        log_file.write(f"training_time_min: {training_time_min:.1f}\n")
        log_file.write(f"n_epochs: {tc.n_epochs}\n")
        log_file.write(f"data_augmentation_loop: {tc.data_augmentation_loop}\n")
        log_file.write(f"recurrent_training: {tc.recurrent_training}\n")
        log_file.write(f"batch_size: {tc.batch_size}\n")
        log_file.write(f"learning_rate_W: {tc.lr_W}\n")
        log_file.write(f"learning_rate: {tc.lr}\n")
        log_file.write(f"learning_rate_embedding: {tc.lr_embedding}\n")
        log_file.write(f"coeff_g_phi_diff: {tc.coeff_g_phi_diff}\n")
        log_file.write(f"coeff_g_phi_norm: {tc.coeff_g_phi_norm}\n")
        log_file.write(f"coeff_g_phi_weight_L1: {tc.coeff_g_phi_weight_L1}\n")
        log_file.write(f"coeff_f_theta_weight_L1: {tc.coeff_f_theta_weight_L1}\n")
        log_file.write(f"coeff_f_theta_weight_L2: {tc.coeff_f_theta_weight_L2}\n")
        log_file.write(f"coeff_W_L1: {tc.coeff_W_L1}\n")
        if field_R2 is not None:
            log_file.write(f"field_R2: {field_R2:.4f}\n")
            log_file.write(f"field_slope: {field_slope:.4f}\n")


# data_train_flyvis_alternate removed — use data_train_flyvis instead
def data_train_flyvis_RNN(config, erase, best_model, device):
    """RNN training with sequential processing through time"""

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model


    warm_up_length = tc.warm_up_length  # e.g., 10
    sequence_length = tc.sequence_length  # e.g., 32
    total_length = warm_up_length + sequence_length

    seed = config.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_dir, logger = create_log_dir(config, erase)

    _logger.info(f"Loading data from {config.dataset}...")
    x_list = []
    y_list = []
    for run in trange(0, tc.n_runs, ncols=100):
        x = np.load(graphs_data_path(config.dataset, f'x_list_{run}.npy'))
        y = np.load(graphs_data_path(config.dataset, f'y_list_{run}.npy'))

        if tc.training_selected_neurons:
            selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
            x = x[:, selected_neuron_ids, :]
            y = y[:, selected_neuron_ids, :]

        x_list.append(x)
        y_list.append(y)

    _logger.info(f'dataset: {len(x_list)} runs, {len(x_list[0])} frames')

    # Normalization
    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]

    if len(valid_distrib) > 0:
        xnorm = 1.5 * torch.std(valid_distrib)
    else:
        xnorm = torch.tensor(1.0, device=device)

    ynorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    _logger.info(f'xnorm: {xnorm.item():.3f}')
    _logger.info(f'ynorm: {ynorm.item():.3f}')
    logger.info(f'xnorm: {xnorm.item():.3f}')
    logger.info(f'ynorm: {ynorm.item():.3f}')

    # Create model
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)
    use_lstm = 'LSTM' in model_config.signal_model_name

    # Count parameters
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_total_params:,}')
    logger.info(f'Total parameters: {n_total_params:,}')

    # Optimizer
    lr = tc.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    _logger.info(f'learning rate: {lr}')
    logger.info(f'learning rate: {lr}')

    _logger.info("starting RNN training...")
    logger.info("Starting RNN training...")

    list_loss = []

    for epoch in range(tc.n_epochs):

        # Number of sequences per epoch
        n_sequences = (sim.n_frames - total_length) // 10 * tc.data_augmentation_loop
        plot_frequency = int(n_sequences // 10) # Sample ~10% of possible sequences
        if epoch == 0:
            _logger.debug(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')
            logger.info(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')

        total_loss = 0
        model.train()

        for seq_idx in trange(n_sequences, ncols=150, desc=f"Epoch {epoch}"):

            optimizer.zero_grad()

            # Sample random sequence
            run = np.random.randint(tc.n_runs)
            k_start = np.random.randint(0, sim.n_frames - total_length)

            # Initialize hidden state to None (GRU will initialize to zeros)
            h = None
            c = None if use_lstm else None

            # Warm-up phase
            with torch.no_grad():
                for t in range(k_start, k_start + warm_up_length):
                    x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                    if use_lstm:
                        _, h, c = model(x, h=h, c=c, return_all=True)
                    else:
                        _, h = model(x, h=h, return_all=True)

            # Prediction phase (compute loss)
            loss = 0
            for t in range(k_start + warm_up_length, k_start + total_length):
                x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                y_true = torch.tensor(y_list[run][t], dtype=torch.float32, device=device)

                # Forward pass
                if use_lstm:
                    y_pred, h, c = model(x, h=h, c=c, return_all=True)
                else:
                    y_pred, h = model(x, h=h, return_all=True)

                # Accumulate loss
                loss += (y_pred - y_true).norm(2)

                # # Truncated BPTT: detach hidden state
                # h = h.detach()

            # Normalize by sequence length
            loss = loss / sequence_length

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if tc.save_all_checkpoints and (seq_idx % plot_frequency == 0) and (seq_idx > 0):
                # Save intermediate model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs-1}_graphs_{epoch}_{seq_idx}.pt'))

        # Epoch statistics
        avg_loss = total_loss / n_sequences
        _logger.info(f"Epoch {epoch}. Loss: {avg_loss:.6f}")
        logger.info(f"Epoch {epoch}. Loss: {avg_loss:.6f}")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs-1}_graphs_{epoch}.pt'))

        list_loss.append(avg_loss)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            _logger.info(f"Learning rate decreased to {param_group['lr']}")
            logger.info(f"Learning rate decreased to {param_group['lr']}")


# INR training moved to graph_trainer_inr.py — re-export for backwards compatibility
from flyvis_gnn.models.graph_trainer_inr import _generate_inr_video, data_train_INR  # noqa: F401


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600,
              ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[],
              rollout_without_noise: bool = False, log_file=None, test_config=None):

    dataset_name = config.dataset
    _logger.info(f"dataset_name: {dataset_name}")
    _logger.info(f"{config.description}")

    if 'fly' in config.dataset:
        # Route to special test (ODE regeneration) for ablation/modification experiments,
        # otherwise use pre-generated test data
        special_modes = ('ablation', 'modified', 'inactivity', 'special')
        if any(m in test_mode for m in special_modes):
            if test_mode == "":
                test_mode = "test_ablation_0"
            data_test_flyvis_special(
                config,
                visualize,
                style,
                verbose,
                best_model,
                step,
                n_rollout_frames,
                test_mode,
                new_params,
                device,
                rollout_without_noise=rollout_without_noise,
                log_file=log_file,
            )
        else:
            data_test_flyvis(
                config,
                best_model=best_model,
                device=device,
                log_file=log_file,
                test_config=test_config,
            )
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")



# Test functions moved to graph_tester.py
from flyvis_gnn.models.graph_tester import data_test_flyvis, data_test_flyvis_special
