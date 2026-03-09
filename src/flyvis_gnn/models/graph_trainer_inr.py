"""INR (Implicit Neural Representation) training for FlyVis.

Trains SIREN or InstantNGP models to learn stimulus/voltage fields
from NeuronTimeSeries data. Extracted from graph_trainer.py.
"""

import os
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FFMpegWriter
from tqdm import trange

from flyvis_gnn.figure_style import default_style
from flyvis_gnn.utils import create_log_dir, graphs_data_path


def _generate_inr_video(gt_np, predict_frame_fn, pos_np, field_name,
                        output_folder, n_frames, step_video=2, fps=10,
                        n_video_frames=800):
    """Generate GT vs Pred MP4 video using FFMpegWriter (streaming).

    Three-panel layout matching data_train: GT hex | Pred hex | rolling traces.
    Includes linear correction (a*pred + b) and per-frame RMSE.

    Args:
        gt_np: (T, N) ground truth numpy array
        predict_frame_fn: callable(frame_idx) -> (N,) numpy array
        pos_np: (N, 2) neuron positions (or None to skip)
        field_name: label for the video
        output_folder: where to write output
        n_frames: total number of frames
        step_video: sample every N-th frame (default 2)
        fps: output video framerate (default 10)
        n_video_frames: number of data frames to include (default 800)
    """
    if pos_np is None:
        print('  no neuron positions — skipping video')
        return

    n_video_frames = min(n_video_frames, n_frames)
    x, y = pos_np[:, 0], pos_np[:, 1]

    # first pass: collect gt and pred for linear fit + color limits
    all_gt, all_pred = [], []
    for k in range(0, n_video_frames, step_video):
        all_gt.append(gt_np[k])
        all_pred.append(predict_frame_fn(k))
    all_gt_flat = np.concatenate(all_gt)
    all_pred_flat = np.concatenate(all_pred)

    # linear fit: gt = a * pred + b
    A_fit = np.vstack([all_pred_flat, np.ones(len(all_pred_flat))]).T
    a_coeff, b_coeff = np.linalg.lstsq(A_fit, all_gt_flat, rcond=None)[0]
    gt_vmin, gt_vmax = float(all_gt_flat.min()), float(all_gt_flat.max())

    # trace neuron selection (10 evenly spaced)
    n_neurons = pos_np.shape[0]
    trace_ids = np.linspace(0, n_neurons - 1, 10, dtype=int)
    win = 200
    offset = 1.25
    hist_t = deque(maxlen=win)
    hist_gt = {i: deque(maxlen=win) for i in trace_ids}
    hist_pred = {i: deque(maxlen=win) for i in trace_ids}

    frame_indices = list(range(0, n_video_frames, step_video))
    print(f'  generating video: {len(frame_indices)} frames, a={a_coeff:.4f} b={b_coeff:.4f}')

    fig = plt.figure(figsize=(12, 4))
    video_path = os.path.join(output_folder, f'{field_name}_gt_vs_pred.mp4')
    metadata = dict(title=f'{field_name} GT vs Pred', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    with writer.saving(fig, video_path, dpi=200):
        error_list = []
        for k in trange(0, n_video_frames, step_video, ncols=100, desc='video'):
            gt_vec = gt_np[k]
            pred_vec = predict_frame_fn(k)
            pred_corrected = a_coeff * pred_vec + b_coeff

            # RMSE
            rmse_frame = float(np.sqrt(((pred_corrected - gt_vec) ** 2).mean()))
            running_rmse = float(np.mean(error_list + [rmse_frame])) if error_list else rmse_frame

            # update rolling traces
            hist_t.append(k)
            for i in trace_ids:
                hist_gt[i].append(gt_vec[i])
                hist_pred[i].append(pred_corrected[i])

            fig.clf()

            # panel 1: GT hex field
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.scatter(x, y, s=256, c=gt_vec, cmap=default_style.cmap,
                        marker='h', vmin=gt_vmin, vmax=gt_vmax)
            ax1.set_axis_off()
            ax1.set_title('ground truth', fontsize=12)

            # panel 2: Pred hex field (corrected)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.scatter(x, y, s=256, c=pred_corrected, cmap=default_style.cmap,
                        marker='h', vmin=gt_vmin, vmax=gt_vmax)
            ax2.set_axis_off()
            ax2.set_title('prediction (corrected)', fontsize=12)

            # panel 3: rolling traces
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_axis_off()
            ax3.set_facecolor('black')
            t_arr = np.arange(len(hist_t))
            for j, i in enumerate(trace_ids):
                y0 = j * offset
                ax3.plot(t_arr, np.array(hist_gt[i]) + y0, color='lime', lw=1.6, alpha=0.95)
                ax3.plot(t_arr, np.array(hist_pred[i]) + y0, color='k', lw=1.2, alpha=0.95)
            ax3.set_xlim(max(0, len(t_arr) - win), len(t_arr))
            ax3.set_ylim(-offset * 0.5, offset * (len(trace_ids) + 0.5))
            ax3.text(0.02, 0.98,
                     f'frame: {k}   RMSE: {rmse_frame:.3f}   avg RMSE: {running_rmse:.3f}   a={a_coeff:.3f} b={b_coeff:.3f}',
                     transform=ax3.transAxes, va='top', ha='left', fontsize=6, color='k')

            plt.tight_layout()
            writer.grab_frame()
            error_list.append(rmse_frame)

    plt.close(fig)
    size_mb = os.path.getsize(video_path) / 1e6
    print(f'  video saved: {video_path} ({size_mb:.1f} MB)')


def data_train_INR(config=None, device=None, total_steps=10000, field_name='stimulus',
                   n_training_frames=0, inr_type=None):
    """Train an INR (SIREN or instantNGP) on a field from x_list_train.

    Loads the specified field from the zarr V3 dataset, trains the INR,
    and produces loss/trace plots plus a results log.

    INR types (auto-detected from config, or set via graph_model.inr_type):
        siren_t:    input=t,        output=n_neurons  (input_size_nnr_f=1)
        siren_txy:  input=(t,x,y),  output=1          (input_size_nnr_f=3)

    Args:
        config: NeuralGraphConfig
        device: torch device
        total_steps: training iterations (default 50000)
        field_name: field to learn from NeuronTimeSeries
                    ('stimulus', 'voltage', 'calcium', 'fluorescence')
        n_training_frames: number of frames to use (0 = use all, >0 = first N frames).
                           Overrides config.training.n_training_frames if >0.
        inr_type: INR architecture to use ('siren_t', 'siren_txy', 'ngp').
                  Overrides config.graph_model.inr_type if not None.
    """
    from scipy.stats import linregress

    from flyvis_gnn.models.Siren_Network import Siren
    from flyvis_gnn.zarr_io import load_simulation_data

    # ANSI colors for R² display
    _GREEN, _YELLOW, _ORANGE, _RED, _RESET = (
        '\033[92m', '\033[93m', '\033[38;5;208m', '\033[91m', '\033[0m')
    def _r2c(v):
        return _GREEN if v > 0.9 else _YELLOW if v > 0.7 else _ORANGE if v > 0.3 else _RED

    sim = config.simulation
    model_config = config.graph_model
    tc = config.training

    log_dir, _ = create_log_dir(config, erase=False)
    output_folder = os.path.join(log_dir, 'tmp_training', f'inr_{field_name}')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)

    # --- load data from zarr V3 ---
    train_path = graphs_data_path(config.dataset, 'x_list_train')
    if os.path.exists(train_path):
        x_ts = load_simulation_data(train_path)
    else:
        print("x_list_train not found, falling back to x_list_0")
        x_ts = load_simulation_data(graphs_data_path(config.dataset, 'x_list_0'))

    field_data = getattr(x_ts, field_name, None)
    if field_data is None:
        raise ValueError(f"field '{field_name}' not found in NeuronTimeSeries "
                         f"(available: voltage, stimulus, calcium, fluorescence)")

    # field_data: (T, N) tensor — restrict to input neurons for stimulus
    field_np = field_data.numpy()
    n_frames, n_neurons_full = field_np.shape
    neuron_pos_full = x_ts.pos.numpy() if x_ts.pos is not None else None

    if field_name == 'stimulus':
        n_input = sim.n_input_neurons
        field_np = field_np[:, :n_input]
        neuron_pos_np = neuron_pos_full[:n_input] if neuron_pos_full is not None else None
        n_neurons = n_input
        print(f'training INR on field "{field_name}" (input neurons only)')
        print(f'  data: {n_frames} frames, {n_neurons} neurons (of {n_neurons_full} total)')
    else:
        neuron_pos_np = neuron_pos_full
        n_neurons = n_neurons_full
        print(f'training INR on field "{field_name}"')
        print(f'  data: {n_frames} frames, {n_neurons} neurons')

    # crop to first N frames if n_training_frames is set
    # function argument overrides config value when >0
    if n_training_frames <= 0:
        n_training_frames = getattr(tc, 'n_training_frames', 0)
    if n_training_frames > 0 and n_training_frames < n_frames:
        field_np = field_np[:n_training_frames]
        n_frames = n_training_frames
        print(f'  cropped to first {n_training_frames} frames')

    # SVD analysis
    from sklearn.utils.extmath import randomized_svd
    n_comp = min(50, min(field_np.shape) - 1)
    _, S, _ = randomized_svd(field_np, n_components=n_comp, random_state=0)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    print(f'  effective rank: 90%={rank_90}, 99%={rank_99}')

    # config parameters
    # function argument overrides config; fall back to auto-detect
    if inr_type is None:
        inr_type = getattr(model_config, 'inr_type', None)
    if inr_type is None:
        input_size_nnr = getattr(model_config, 'input_size_nnr_f', 1)
        output_size_nnr = getattr(model_config, 'output_size_nnr_f', None)
        if input_size_nnr == 3 and output_size_nnr == 1:
            inr_type = 'siren_txy'
        else:
            inr_type = 'siren_t'
    # normalize enum to plain string for consistent filenames / comparisons
    inr_type = str(inr_type).rsplit('.', 1)[-1].lower()
    # clean old comparison PNGs from previous runs
    for old_png in [f for f in os.listdir(output_folder) if f.endswith('.png')]:
        os.remove(os.path.join(output_folder, old_png))
    hidden_dim = getattr(model_config, 'hidden_dim_nnr_f', 1024)
    n_layers = getattr(model_config, 'n_layers_nnr_f', 3)
    omega_f = getattr(model_config, 'omega_f', 1024)
    omega_f_learning = getattr(model_config, 'omega_f_learning', False)
    t_period = getattr(model_config, 'nnr_f_T_period', n_frames) / (2 * np.pi)
    xy_period = getattr(model_config, 'nnr_f_xy_period', 1.0) / (2 * np.pi)
    batch_size = getattr(tc, 'inr_batch_size', 8)
    learning_rate = getattr(tc, 'learning_rate_NNR_f', 1e-6)

    # --- build model ---
    if inr_type == 'siren_t':
        input_dim, output_dim = 1, n_neurons
    elif inr_type == 'siren_txy':
        input_dim, output_dim = 3, 1  # (t, x, y) -> scalar
    elif inr_type == 'ngp':
        input_dim = getattr(model_config, 'input_size_nnr_f', 1)
        output_dim = getattr(model_config, 'output_size_nnr_f', n_neurons)
    else:
        raise ValueError(f"unknown inr_type: {inr_type}")

    if inr_type == 'ngp':
        try:
            from cell_gnn.models.HashEncoding_Network import HashEncodingMLP
        except ImportError:
            raise ImportError("HashEncodingMLP requires cell_gnn package (tinycudann)")
        nnr_f = HashEncodingMLP(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            n_levels=getattr(model_config, 'ngp_n_levels', 24),
            n_features_per_level=getattr(model_config, 'ngp_n_features_per_level', 2),
            log2_hashmap_size=getattr(model_config, 'ngp_log2_hashmap_size', 22),
            base_resolution=getattr(model_config, 'ngp_base_resolution', 16),
            per_level_scale=getattr(model_config, 'ngp_per_level_scale', 1.4),
            n_neurons=getattr(model_config, 'ngp_n_neurons', 128),
            n_hidden_layers=getattr(model_config, 'ngp_n_hidden_layers', 4),
            output_activation='none',
        ).to(device)
    else:
        nnr_f = Siren(
            in_features=input_dim,
            hidden_features=hidden_dim,
            hidden_layers=n_layers,
            out_features=output_dim,
            outermost_linear=True,
            first_omega_0=omega_f,
            hidden_omega_0=omega_f,
            learnable_omega=omega_f_learning,
        ).to(device)

    total_params = sum(p.numel() for p in nnr_f.parameters())
    data_dims = n_frames * n_neurons
    print(f'  INR type: {inr_type}, params: {total_params:,}, '
          f'compression: {data_dims / total_params:.1f}x')

    # --- optimizer ---
    omega_params = [p for name, p in nnr_f.named_parameters() if 'omega' in name]
    other_params = [p for name, p in nnr_f.named_parameters() if 'omega' not in name]
    lr_omega = getattr(tc, 'learning_rate_omega_f', learning_rate)
    if omega_params and omega_f_learning:
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': omega_params, 'lr': lr_omega},
        ])
    else:
        optim = torch.optim.Adam(nnr_f.parameters(), lr=learning_rate)

    # prepare tensors
    ground_truth = torch.tensor(field_np, dtype=torch.float32, device=device)
    if inr_type == 'siren_txy':
        neuron_pos = torch.tensor(neuron_pos_np / xy_period, dtype=torch.float32, device=device)

    # --- predict helper for siren_txy (single frame) ---
    def _predict_frame_txy(frame_idx):
        """Predict a single frame for siren_txy."""
        with torch.no_grad():
            t_val = torch.full((n_neurons, 1), frame_idx / t_period, device=device)
            inp = torch.cat([t_val, neuron_pos], dim=1)
            return nnr_f(inp).squeeze()

    # --- predict sampled frames for R² ---
    def _predict_sampled(n_sample=200):
        """Predict a random subset of frames for fast R² estimation."""
        sample_ids = np.linspace(0, n_frames - 1, n_sample, dtype=int)
        gt_sample = ground_truth[sample_ids]
        with torch.no_grad():
            if inr_type == 'siren_txy':
                preds = []
                for t_idx in sample_ids:
                    preds.append(_predict_frame_txy(t_idx))
                pred_sample = torch.stack(preds, dim=0)
            elif inr_type in ('siren_t', 'ngp'):
                t_batch = torch.tensor(sample_ids, dtype=torch.float32, device=device).unsqueeze(1) / t_period
                pred_sample = nnr_f(t_batch)
        return gt_sample.cpu().numpy(), pred_sample.cpu().numpy()

    # --- predict all frames (used for final evaluation) ---
    def _predict_all():
        with torch.no_grad():
            if inr_type in ('siren_t', 'ngp'):
                t_all = torch.arange(n_frames, dtype=torch.float32, device=device).unsqueeze(1) / t_period
                return nnr_f(t_all)
            elif inr_type == 'siren_txy':
                results = []
                for t_idx in range(n_frames):
                    results.append(_predict_frame_txy(t_idx))
                return torch.stack(results, dim=0)

    # --- training loop ---
    loss_list = []
    report_interval = 1000
    viz_interval = 5000
    last_r2 = 0.0
    best_r2 = 0.0
    t_start = time.time()

    print(f'  training for {total_steps} steps, batch_size={batch_size}, lr={learning_rate}')
    print(f'  saving plot every {viz_interval} steps, R² eval every {report_interval} steps')

    pbar = trange(total_steps + 1, ncols=120, desc=f'INR {field_name}')
    for step in pbar:
        optim.zero_grad()
        sample_ids = np.random.choice(n_frames, batch_size, replace=(batch_size > n_frames))
        gt_batch = ground_truth[sample_ids]

        if inr_type == 'siren_t':
            t_batch = torch.tensor(sample_ids, dtype=torch.float32, device=device).unsqueeze(1) / t_period
            pred = nnr_f(t_batch)
            loss = F.mse_loss(pred, gt_batch)

        elif inr_type == 'siren_txy':
            t_norm = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_neurons, 1)
            pos_expanded = neuron_pos[None, :, :].expand(batch_size, n_neurons, 2)
            inp = torch.cat([t_expanded, pos_expanded], dim=2).reshape(batch_size * n_neurons, 3)
            gt_flat = gt_batch.reshape(batch_size * n_neurons)
            pred = nnr_f(inp).squeeze()
            loss = F.mse_loss(pred, gt_flat)

        elif inr_type == 'ngp':
            t_batch = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
            pred = nnr_f(t_batch)
            rel_l2 = (pred - gt_batch.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)
            loss = rel_l2.mean()

        # omega L2 regularization
        coeff_omega_L2 = getattr(tc, 'coeff_omega_f_L2', 0.0)
        if omega_f_learning and coeff_omega_L2 > 0 and hasattr(nnr_f, 'get_omega_L2_loss'):
            loss = loss + coeff_omega_L2 * nnr_f.get_omega_L2_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nnr_f.parameters(), max_norm=1.0)
        optim.step()
        loss_list.append(loss.item())

        # R² evaluation (sampled for speed)
        if step > 0 and step % report_interval == 0:
            gt_s, pred_s = _predict_sampled(n_sample=200)
            _, _, r_value, _, _ = linregress(gt_s.reshape(-1), pred_s.reshape(-1))
            last_r2 = r_value ** 2
            if last_r2 > best_r2:
                best_r2 = last_r2

        if step % 1000 == 0:
            c = _r2c(last_r2)
            pbar.set_postfix_str(f'loss={loss.item():.6f} {c}R²={last_r2:.4f} best={best_r2:.4f}{_RESET}')

        # visualization: hex comparison at frame n_frames//2
        if step > 0 and step % viz_interval == 0 and neuron_pos_np is not None:
            mid_fr = n_frames // 2
            with torch.no_grad():
                if inr_type == 'siren_txy':
                    t_val = torch.full((n_neurons, 1), mid_fr / t_period, device=device)
                    inp = torch.cat([t_val, neuron_pos], dim=1)
                    pred_frame = nnr_f(inp).squeeze().cpu().numpy()
                else:
                    pred_all = _predict_all()
                    pred_frame = pred_all.cpu().numpy()[mid_fr]
            gt_frame = field_np[mid_fr]
            vmin, vmax = np.percentile(gt_frame, 2), np.percentile(gt_frame, 98)
            fig_cmp, (ax_gt, ax_pr) = plt.subplots(1, 2, figsize=(10, 5))
            px, py = neuron_pos_np[:, 0], neuron_pos_np[:, 1]
            ax_gt.scatter(px, py, s=256, c=gt_frame, cmap='viridis',
                          marker='h', vmin=vmin, vmax=vmax)
            ax_gt.set_title('ground truth', fontsize=12)
            ax_gt.set_axis_off()
            ax_pr.scatter(px, py, s=256, c=pred_frame, cmap='viridis',
                          marker='h', vmin=vmin, vmax=vmax)
            ax_pr.set_title('prediction', fontsize=12)
            ax_pr.set_axis_off()
            fig_cmp.suptitle(f'{field_name}  step {step}  R²={last_r2:.4f}', fontsize=11)
            fig_cmp.tight_layout()
            cmp_path = f"{output_folder}/{inr_type}_comparison_{step}.png"
            os.makedirs(os.path.dirname(cmp_path), exist_ok=True)
            fig_cmp.savefig(cmp_path, dpi=150)
            plt.close(fig_cmp)

    # --- final evaluation (sampled) ---
    elapsed = time.time() - t_start
    gt_s, pred_s = _predict_sampled(n_sample=500)
    final_mse = np.mean((gt_s - pred_s) ** 2)
    _, _, r_value, _, _ = linregress(gt_s.reshape(-1), pred_s.reshape(-1))
    final_r2 = r_value ** 2
    if final_r2 > best_r2:
        best_r2 = final_r2
    r2_drop = best_r2 - final_r2

    print(f'  training complete: {elapsed / 60:.1f} min')
    print(f'  final MSE: {final_mse:.6e}, R²: {final_r2:.6f}, best R²: {best_r2:.6f}, drop: {r2_drop:.6f}')
    if hasattr(nnr_f, 'get_omegas'):
        print(f'  final omegas: {nnr_f.get_omegas()}')

    # save model
    model_path = os.path.join(log_dir, 'models', f'inr_{field_name}.pt')
    torch.save(nnr_f.state_dict(), model_path)
    print(f'  model saved to {model_path}')

    # results log
    results_path = os.path.join(output_folder, 'results.log')
    with open(results_path, 'w') as f:
        f.write(f'field_name: {field_name}\n')
        f.write(f'inr_type: {inr_type}\n')
        f.write(f'final_mse: {final_mse:.6e}\n')
        f.write(f'final_r2: {final_r2:.6f}\n')
        f.write(f'best_r2: {best_r2:.6f}\n')
        f.write(f'r2_drop: {r2_drop:.6f}\n')
        f.write(f'n_neurons: {n_neurons}\n')
        f.write(f'n_frames: {n_frames}\n')
        f.write(f'total_steps: {total_steps}\n')
        f.write(f'total_params: {total_params}\n')
        f.write(f'training_time_min: {elapsed / 60:.1f}\n')
        f.write(f'rank_90: {rank_90}\n')
        f.write(f'rank_99: {rank_99}\n')
    print(f'  results written to {results_path}')

    # --- generate GT vs Pred video ---
    def _predict_frame_np(frame_idx):
        """Predict one frame and return numpy array."""
        if inr_type == 'siren_txy':
            return _predict_frame_txy(frame_idx).cpu().numpy()
        elif inr_type in ('siren_t', 'ngp'):
            with torch.no_grad():
                t_val = torch.tensor([[frame_idx / t_period]], dtype=torch.float32, device=device)
                return nnr_f(t_val).squeeze().cpu().numpy()

    _generate_inr_video(field_np, _predict_frame_np, neuron_pos_np, field_name,
                        output_folder, n_frames=n_frames)

    return nnr_f, loss_list
