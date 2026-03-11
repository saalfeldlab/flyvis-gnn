import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

# Optional imports (not available in flyvis-gnn spinoff)
try:
    from flyvis_gnn.data_loaders import load_wormvae_data, load_zebrafish_data
except ImportError:
    load_wormvae_data = None
    load_zebrafish_data = None
from flyvis_gnn.figure_style import dark_style, default_style
from flyvis_gnn.log import get_logger
from flyvis_gnn.neuron_state import NeuronState
from flyvis_gnn.plot import (
    plot_activity_traces,
    plot_selected_neuron_traces,
    plot_spatial_activity_grid,
)
from flyvis_gnn.zarr_io import ZarrArrayWriter, ZarrSimulationWriterV3

try:
    from flyvis_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset
except ImportError:
    AugmentedVideoDataset = None
    CombinedVideoDataset = None
import os

from tqdm import tqdm

from flyvis_gnn.generators.utils import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    generate_compressed_video_mp4,
    get_equidistant_points,
    greedy_blue_mask,
    mseq_bits,
)
from flyvis_gnn.utils import get_datavis_root_dir, graphs_data_path, to_numpy

logger = get_logger(__name__)


def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    best_model=None,
    device=None,
    save=True,
    log_file=None,
):

    logger.info(f"dataset: {config.dataset}")

    if (os.path.isdir(graphs_data_path(config.dataset, "x_list_train"))
        or os.path.isfile(graphs_data_path(config.dataset, "x_list_0.npy"))
        or os.path.isfile(graphs_data_path(config.dataset, "x_list_0.pt"))
    ):
        logger.warning("watch out: data already generated")
        # return

    if config.data_folder_name != "none":
        generate_from_data(config=config, device=device, visualize=visualize, style=style, step=step)
    else:
        data_generate_fly_voltage(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            save=save,
        )

    default_style.apply_globally()


def generate_from_data(config, device, visualize=True, step=None, cmap=None, style=None):
    data_folder_name = config.data_folder_name

    if "wormvae" in data_folder_name:
        load_wormvae_data(config, device, visualize, step)
    elif "NeuroPAL" in data_folder_name:
        # load_neuropal_data(config, device, visualize, step)  # TODO: Function not yet implemented
        raise NotImplementedError("NeuroPAL data loading not yet implemented")
    elif 'Zapbench' in data_folder_name:
        load_zebrafish_data(config, device, visualize, step, cmap, style)
    else:
        raise ValueError(f"Unknown data folder name {data_folder_name}")

def _plot_sequence_preview(sequences, hex_x, hex_y, title, save_path, fig_style,
                           metadata=None):
    """Plot first frame of first N sequences as hex maps.

    Args:
        metadata: optional list of (name, flip_ax, n_rot) tuples per sequence.
    """
    try:
        # Compute cumulative frame offsets from actual sequence lengths
        cum_offsets = []
        offset = 0
        for seq in sequences:
            n_fr = seq["lum"].shape[0]
            cum_offsets.append((offset, offset + n_fr))
            offset += n_fr

        n_cols = 8
        n_preview = min(n_cols * 8, len(sequences))
        n_rows = (n_preview + n_cols - 1) // n_cols
        fig_preview, axes_preview = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
        axes_preview = np.atleast_2d(axes_preview)
        for i in range(n_preview):
            row, col = divmod(i, n_cols)
            lum = sequences[i]["lum"]
            vals = lum[0].squeeze().cpu().numpy() if isinstance(lum, torch.Tensor) else lum[0].squeeze()
            start, stop = cum_offsets[i]
            ax = axes_preview[row, col]
            ax.scatter(hex_x, hex_y, c=vals,
                       s=fig_style.hex_stimulus_marker_size,
                       marker=fig_style.hex_marker,
                       cmap=fig_style.cmap,
                       vmin=fig_style.hex_stimulus_range[0],
                       vmax=fig_style.hex_stimulus_range[1],
                       alpha=1.0, linewidths=0)
            ax.set_facecolor(fig_style.background)
            if metadata is not None and i < len(metadata):
                name, flip, rot = metadata[i][:3]
                short = str(name).split('_split_')[0].split('sequence_')[-1] if 'sequence_' in str(name) else str(name)
                ax.set_title(f"{short}\nf{flip} r{rot} [{start}:{stop}]", fontsize=4)
            else:
                ax.set_title(f"seq {i} [{start}:{stop}]", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in axes_preview.flat:
            if not ax.has_data():
                ax.set_visible(False)
        fig_preview.suptitle(title, fontsize=9)
        fig_preview.tight_layout()
        fig_preview.savefig(save_path, dpi=200)
        plt.close(fig_preview)
        logger.info(f"saved: {save_path}")
    except Exception as e:
        logger.warning(f"could not save sequence preview: {e}")
        import traceback
        traceback.print_exc()
        plt.close("all")


def _run_ode_generation(stimulus_sequences, net, pde, x, edge_index, initial_state,
                        sim, x_writer, y_writer, target_frames, num_passes,
                        n_neurons, device, to_numpy_fn,
                        visualize=False, run=0, run_vizualized=0, step=5,
                        id_fig_start=0, it_start=0, fig_style=None,
                        config=None, davis_dataset=None,
                        X1=None, u_coords=None, v_coords=None):
    """Run ODE simulation over stimulus sequences, writing frames to zarr.

    This is the inner loop extracted so it can be called for both train and test.
    Returns (it, id_fig) — the final frame counter and figure counter.
    """
    it = it_start
    id_fig = id_fig_start

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    n_columns = sim.n_input_neurons // 8

    # Mixed sequence setup
    mixed_types_list = None
    if "mixed" in sim.visual_input_type:
        mixed_types_list = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        sintel_iter = iter(stimulus_sequences)
        davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0

    with torch.no_grad():
        for pass_num in range(num_passes):
            for data_idx, data in enumerate(tqdm(stimulus_sequences, desc="processing stimulus data", ncols=100)):
                if sim.simulation_initial_state:
                    x.voltage[:] = initial_state
                    if sim.only_noise_visual_input > 0:
                        x.stimulus[:sim.n_input_neurons] = torch.clamp(torch.relu(
                            0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32,
                                             device=device) * sim.only_noise_visual_input / 2), 0, 1)

                sequences = data["lum"]

                if "flash" in sim.visual_input_type:
                    flash_duration_options = [1, 2, 5]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]
                    flash_intensity = torch.abs(torch.rand(sim.n_input_neurons, device=device) * 0.5 + 0.5)

                if mixed_types_list is not None:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types_list[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_sequences)
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
                                davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0

                if "flash" in sim.visual_input_type:
                    sequence_length = 60
                else:
                    sequence_length = sequences.shape[0]

                for frame_id in range(sequence_length):
                    if "flash" in sim.visual_input_type:
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)
                        x.stimulus[:] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x.stimulus[:sim.n_input_neurons] = flash_intensity
                    elif mixed_types_list is not None:
                        current_type = mixed_types_list[mixed_current_type]
                        if current_type == "blank":
                            x.stimulus[:] = 0
                        elif current_type == "noise":
                            x.stimulus[:sim.n_input_neurons] = torch.relu(
                                0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device) * 0.5)
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x.stimulus[:] = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=sim.seed
                            )
                            base = mseq_bits(p=8, seed=sim.seed).astype(np.float32)
                            rng = np.random.RandomState(sim.seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(device, dtype=torch.float32)
                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0

                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]
                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            tile_labels_np, col_centers = compute_column_labels(u_coords, v_coords, n_columns, seed=sim.seed)
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform
                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [set(node_params.Where((D[i] > 0) & (D[i] <= radius))[0].tolist()) for i in
                                       range(len(col_centers))]

                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=device)
                            rng = np.random.RandomState(sim.seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)
                                vals = node_params.Where(mask, 1.0, -1.0).astype(np.float32)
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(device, dtype=torch.float32)

                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
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
                                    x.stimulus[:] = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus[:] = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[:sim.n_input_neurons] = x.stimulus[:sim.n_input_neurons] + torch.randn(sim.n_input_neurons,
                                                                                                  dtype=torch.float32,
                                                                                                  device=device) * sim.noise_visual_input

                    y = pde(x, edge_index, has_field=False)
                    prev_calcium = x.calcium.clone()

                    # Generate measurement noise for this timestep
                    if sim.measurement_noise_level > 0:
                        x.noise = torch.randn(n_neurons, dtype=torch.float32, device=device) * sim.measurement_noise_level
                    else:
                        x.noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)

                    x_writer.append_state(x)

                    dv = y.squeeze()
                    if sim.noise_model_level > 0:
                        x.voltage = x.voltage + sim.delta_t * dv + torch.randn(n_neurons, dtype=torch.float32, device=device) * sim.noise_model_level
                    else:
                        x.voltage = x.voltage + sim.delta_t * dv

                    if sim.calcium_type == "leaky":
                        if sim.calcium_activation == "softplus":
                            s = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            s = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            s = 1 + torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            s = x.voltage.clone()

                        x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + s)
                        x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta
                        y = ((x.calcium - prev_calcium) / sim.delta_t).unsqueeze(-1)

                    y_writer.append(to_numpy_fn(y.clone().detach()))

                    if (visualize & (run == run_vizualized) & (it > 0) & (it % step == 0) & (it <= 50 * step)):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy_fn(X1),
                            voltages=to_numpy_fn(x.voltage),
                            stimulus=to_numpy_fn(x.stimulus[:sim.n_input_neurons]),
                            neuron_types=to_numpy_fn(x.neuron_type).astype(int),
                            output_path=graphs_data_path(config.dataset, "Fig", f"Fig_{run}_{num}.png"),
                            calcium=to_numpy_fn(x.calcium) if sim.calcium_type != "none" else None,
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

    return it, id_fig


def _compute_noisy_derivatives(config, sim, n_neurons, split='train'):
    """Compute noisy derivatives from saved clean derivatives and noise.

    noisy_y[t] = y_clean[t] + (noise[t+1] - noise[t]) / dt
    Last frame uses clean derivative (no future noise available).
    """
    from flyvis_gnn.utils import graphs_data_path
    from flyvis_gnn.zarr_io import ZarrArrayWriter, load_raw_array, load_simulation_data

    y_clean = load_raw_array(graphs_data_path(config.dataset, f"y_list_{split}"))  # (T, N, 1)
    noise_ts = load_simulation_data(
        graphs_data_path(config.dataset, f"x_list_{split}"), fields=['noise']
    )
    noise = noise_ts.noise.numpy()  # (T, N)

    # Compute noise derivative: (noise[t+1] - noise[t]) / dt
    noise_diff = np.zeros_like(noise)
    noise_diff[:-1] = (noise[1:] - noise[:-1]) / sim.delta_t  # last frame: 0

    noisy_y = y_clean + noise_diff[:, :, np.newaxis]  # broadcast to (T, N, 1)

    # Temporal smoothing of noisy derivatives (reduces derivative noise by sqrt(window))
    window = sim.derivative_smoothing_window
    if window > 1:
        from scipy.ndimage import uniform_filter1d
        # Apply centered moving average along time axis (axis=0)
        # mode='nearest' pads boundaries with edge values
        noisy_y = uniform_filter1d(noisy_y, size=window, axis=0, mode='nearest')
        logger.debug(f"  applied derivative smoothing: window={window} (noise reduction ~{1/np.sqrt(window):.2f}x)")

    noisy_y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, f"noisy_y_list_{split}"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )
    for t in range(noisy_y.shape[0]):
        noisy_y_writer.append(noisy_y[t])
    noisy_y_writer.finalize()
    logger.info(f"computed noisy derivatives for {split}: {noisy_y.shape[0]} frames "
                f"(measurement_noise_level={sim.measurement_noise_level})")


def data_generate_fly_voltage(config, visualize=True, run_vizualized=0, style="color", erase=False, step=5, device=None,
                              save=True):

    fig_style = dark_style if "black" in style else default_style
    fig_style.apply_globally()

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    if sim.seed != 42:
        torch.random.manual_seed(sim.seed)
        np.random.seed(sim.seed)  # Ensure numpy random state is also seeded for reproducibility

    n_frames = sim.n_frames
    n_neurons = sim.n_neurons

    logger.info(f"generating data ... {model_config.signal_model_name}  dynamics_noise: {sim.noise_model_level}  measurement_noise: {sim.measurement_noise_level}  seed: {sim.seed}")

    run = 0

    os.makedirs(graphs_data_path("fly"), exist_ok=True)
    folder = graphs_data_path(config.dataset) + "/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(graphs_data_path(config.dataset, "Fig"), exist_ok=True)
    files = glob.glob(graphs_data_path(config.dataset, "Fig", "*"))
    for f in files:
        os.remove(f)

    extent = 8

    # flyvis.__init__ sets root logger to INFO via basicConfig — restore to WARNING
    import logging

    from flyvis import Network, NetworkView
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    from flyvis_gnn.generators.flyvis_ode import (
        FlyVisODE,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )
    from flyvis_gnn.generators.ode_params import FlyVisODEParams, get_ode_params_class
    from flyvis_gnn.utils import setup_flyvis_model_path

    logging.getLogger().setLevel(logging.WARNING)
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
            "temporal_split": False,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False,
            "skip_short_videos": sim.skip_short_videos,
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

    # Initialize the ground-truth flyvis network from a pre-trained checkpoint.
    # This loads the biological connectome (neuron types, synaptic weights, time constants)
    # from the flyvis library, using ensemble_id/model_id to select a specific trained model.
    # The network is then used as the "simulator" to generate voltage traces via its PDE dynamics.
    # Suppress noisy flyvis "epe not in ... Falling back to loss" warning
    import logging as _logging
    _logging.getLogger("flyvis.utils.logging_utils").setLevel(_logging.ERROR)
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    # Extract ground-truth parameters: time constants (tau), resting potentials (V_rest),
    # and effective synaptic weights (strength * count * sign) for PDE simulation.
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
            # Per pre-synaptic neuron: add a proportional number of false targets
            # Compute out-degree per source neuron
            from collections import Counter
            out_degree = Counter(src_np.tolist())
            total_real = edge_index.shape[1]
            ratio = sim.n_extra_null_edges / total_real

            # Build per-neuron target sets for fast lookup
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
                # Sample false targets not already connected and not self
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
            # Random: sample uniformly across the full matrix
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
            logger.info(f"Total extra edges added: {len(extra_edges)}")

    # Edge ablation: zero out a fraction of edge weights before ODE simulation
    ablation_mask = None
    if sim.ablation_ratio > 0:
        rng = np.random.RandomState(sim.ablation_seed)
        n_edges = edge_index.shape[1]
        n_ablate = int(np.round(n_edges * sim.ablation_ratio))
        ablate_indices = rng.choice(n_edges, size=n_ablate, replace=False)
        ablation_mask = torch.ones(n_edges, dtype=torch.bool, device=device)
        ablation_mask[ablate_indices] = False
        ode_params.W[~ablation_mask] = 0.0
        logger.info(f"ablated {n_ablate}/{n_edges} edges ({sim.ablation_ratio*100:.0f}%)")

    pde = FlyVisODE(ode_params=ode_params, g_phi=torch.nn.functional.relu, params=sim.params,
                    model_type=model_config.signal_model_name, n_neuron_types=sim.n_neuron_types, device=device)

    # Edge removal: drop a fraction of edges before saving
    # (simulation already ran with the full graph)
    if sim.edge_removal_ratio > 0:
        # Save full edges first (for reference / analysis)
        if save:
            torch.save(ode_params.W.clone(), graphs_data_path(config.dataset, "weights_full.pt"))
            torch.save(edge_index.clone(), graphs_data_path(config.dataset, "edge_index_full.pt"))

        rng_rm = np.random.RandomState(sim.edge_removal_seed)
        n_total = edge_index.shape[1]
        removal_mode = getattr(sim, 'edge_removal_mode', 'random')
        logger.info(f"edge removal mode: {removal_mode}, ratio: {sim.edge_removal_ratio}")

        if removal_mode == 'per_column':
            # Remove a consistent fraction of outgoing edges per pre-synaptic neuron
            src_np = edge_index[0].cpu().numpy()
            keep_mask = np.ones(n_total, dtype=bool)
            for source in np.unique(src_np):
                source_edges = node_params.Where(src_np == source)[0]
                n_remove = max(1, int(round(len(source_edges) * sim.edge_removal_ratio)))
                if n_remove >= len(source_edges):
                    n_remove = len(source_edges) - 1  # keep at least one
                remove_idx = rng_rm.choice(source_edges, n_remove, replace=False)
                keep_mask[remove_idx] = False
            kept_indices = node_params.Where(keep_mask)[0]
        else:
            # Random removal across the full edge set
            n_keep = int(n_total * (1 - sim.edge_removal_ratio))
            kept_indices = np.sort(rng_rm.choice(n_total, n_keep, replace=False))

        edge_index = edge_index[:, kept_indices]
        ode_params.edge_index = edge_index
        ode_params.W = ode_params.W[kept_indices]
        logger.info(f"edge removal: kept {len(kept_indices)}/{n_total} edges "
                     f"({(1 - len(kept_indices)/n_total)*100:.1f}% removed)")
        if save:
            torch.save(torch.tensor(kept_indices),
                        graphs_data_path(config.dataset, "kept_edge_indices.pt"))

    if save:
        ode_params.save(folder)
        if ablation_mask is not None:
            torch.save(ablation_mask, graphs_data_path(config.dataset, "ablation_mask.pt"))

    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    _ , node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device)

    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0))]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=sim.delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    # init neuron state x

    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=X1,
        voltage=initial_state,
        stimulus=net.stimulus().squeeze(),
        group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
        neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
        calcium=_init_calcium,
        fluorescence=sim.calcium_alpha * _init_calcium + sim.calcium_beta,
        noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )

    # --- Subdirectory-level train/test split ---
    # arg_df is aligned with cached_sequences (shuffle applied to both in _build).
    # Split by original_index so all augmentations of the same base video stay together.
    df = stimulus_dataset.arg_df
    original_indices = df['original_index'].values
    unique_videos = sorted(set(original_indices))
    n_train_vids = int(len(unique_videos) * 0.8)
    train_video_set = set(unique_videos[:n_train_vids])
    test_video_set = set(unique_videos[n_train_vids:])

    train_indices = [i for i, oi in enumerate(original_indices) if oi in train_video_set]
    test_indices = [i for i, oi in enumerate(original_indices) if oi in test_video_set]

    # Extract the actual video subdirectory names for logging
    train_video_names = sorted(set(df.iloc[train_indices]['name'].values))
    test_video_names = sorted(set(df.iloc[test_indices]['name'].values))

    # Verify exclusivity
    train_name_set = set(train_video_names)
    test_name_set = set(test_video_names)
    overlap = train_name_set & test_name_set
    assert len(overlap) == 0, f"TRAIN/TEST OVERLAP: {overlap}"
    logger.info(f"subdirectory split: {n_train_vids} train / {len(unique_videos) - n_train_vids} test videos"
                f"  ({len(train_indices)} train seqs, {len(test_indices)} test seqs)")
    logger.info(f"overlap: {overlap} (must be empty)")

    # Build sequences lists for ODE generation
    train_sequences = [stimulus_dataset[i] for i in train_indices]
    test_sequences = [stimulus_dataset[i] for i in test_indices]

    # Build metadata labels for preview plots (name, flip_ax, n_rot)
    train_meta = [
        (df.iloc[idx]['name'], df.iloc[idx]['flip_ax'], df.iloc[idx]['n_rot'])
        for idx in train_indices
    ]
    test_meta = [
        (df.iloc[idx]['name'], df.iloc[idx]['flip_ax'], df.iloc[idx]['n_rot'])
        for idx in test_indices
    ]

    # Plot preview for train and test splits
    frames_per_sequence = 35
    n_hexals = stimulus_dataset[0]["lum"].shape[-1]
    hex_x = x_coords[:n_hexals]
    hex_y = y_coords[:n_hexals]
    _plot_sequence_preview(train_sequences, hex_x, hex_y,
                           f"TRAIN: {len(train_sequences)} seqs from {n_train_vids} videos",
                           os.path.join(folder, "shuffle_first_frames_train.png"), fig_style,
                           metadata=train_meta)
    _plot_sequence_preview(test_sequences, hex_x, hex_y,
                           f"TEST: {len(test_sequences)} seqs from {len(test_video_set)} videos",
                           os.path.join(folder, "shuffle_first_frames_test.png"), fig_style,
                           metadata=test_meta)

    # --- Generate TRAIN split ---
    total_frames_per_pass = len(train_sequences) * frames_per_sequence

    if n_frames == 0:
        num_passes_needed = 1
        target_frames = float('inf')
        logger.info(f"n_frames=0 mode: single pass through {len(train_sequences)} train sequences")
    else:
        target_frames = n_frames
        num_passes_needed = (target_frames // total_frames_per_pass) + 1

    logger.info(f"generating TRAIN data ({target_frames} frames from {len(train_sequences)} sequences)...")

    x_writer = ZarrSimulationWriterV3(
        path=graphs_data_path(config.dataset, "x_list_train"),
        n_neurons=n_neurons,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, "y_list_train"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )

    it, id_fig = _run_ode_generation(
        stimulus_sequences=train_sequences, net=net, pde=pde, x=x,
        edge_index=edge_index, initial_state=initial_state, sim=sim,
        x_writer=x_writer, y_writer=y_writer,
        target_frames=target_frames, num_passes=num_passes_needed,
        n_neurons=n_neurons, device=device, to_numpy_fn=to_numpy,
        visualize=visualize, run=run, run_vizualized=run_vizualized,
        step=step, id_fig_start=0, it_start=sim.start_frame,
        fig_style=fig_style, config=config, davis_dataset=davis_dataset,
        X1=X1, u_coords=u_coords, v_coords=v_coords,
    )

    n_frames_train = x_writer.finalize()
    y_writer.finalize()
    logger.info(f"generated {n_frames_train} TRAIN frames (saved as .zarr)")

    # --- Compute noisy derivatives for TRAIN split ---
    if sim.measurement_noise_level > 0:
        _compute_noisy_derivatives(config, sim, n_neurons, split='train')

    # --- Generate TEST split ---
    # Reset neural state to avoid train→test leakage
    x.voltage[:] = initial_state
    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x.calcium = _init_calcium
    x.fluorescence = sim.calcium_alpha * _init_calcium + sim.calcium_beta

    # Test: single pass through test sequences
    test_target = len(test_sequences) * frames_per_sequence
    logger.info(f"generating TEST data ({test_target} frames from {len(test_sequences)} sequences)...")

    x_writer = ZarrSimulationWriterV3(
        path=graphs_data_path(config.dataset, "x_list_test"),
        n_neurons=n_neurons,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, "y_list_test"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )

    _run_ode_generation(
        stimulus_sequences=test_sequences, net=net, pde=pde, x=x,
        edge_index=edge_index, initial_state=initial_state, sim=sim,
        x_writer=x_writer, y_writer=y_writer,
        target_frames=float('inf'), num_passes=1,  # single pass, all test sequences
        n_neurons=n_neurons, device=device, to_numpy_fn=to_numpy,
        visualize=False, run=run, run_vizualized=run_vizualized,
        step=step, id_fig_start=id_fig, it_start=0,
        fig_style=fig_style, config=config, davis_dataset=davis_dataset,
        X1=X1, u_coords=u_coords, v_coords=v_coords,
    )

    n_frames_test = x_writer.finalize()
    y_writer.finalize()
    logger.info(f"generated {n_frames_test} TEST frames (saved as .zarr)")

    # --- Compute noisy derivatives for TEST split ---
    if sim.measurement_noise_level > 0:
        _compute_noisy_derivatives(config, sim, n_neurons, split='test')

    # restore gradient computation now (before any early-return paths)
    torch.set_grad_enabled(True)

    # --- Always run diagnostics after data generation ---
    from flyvis_gnn.zarr_io import load_raw_array, load_simulation_data
    x_ts = load_simulation_data(graphs_data_path(config.dataset, "x_list_train"))
    y_list = load_raw_array(graphs_data_path(config.dataset, "y_list_train"))

    # Compute ranks (used in kinographs and traces)
    logger.info('computing effective rank ...')
    from sklearn.utils.extmath import randomized_svd
    activity_full = x_ts.voltage.numpy()  # (n_frames, n_neurons)
    n_comp = min(50, min(activity_full.shape) - 1)
    _, S_act, _ = randomized_svd(activity_full, n_components=n_comp, random_state=0)
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = int(np.searchsorted(cumvar_act, 0.90) + 1)
    rank_99_act = int(np.searchsorted(cumvar_act, 0.99) + 1)

    input_for_svd = x_ts.stimulus[:, :sim.n_input_neurons].numpy()
    n_comp_input = min(50, min(input_for_svd.shape) - 1)
    _, S_inp, _ = randomized_svd(input_for_svd, n_components=n_comp_input, random_state=0)
    cumvar_inp = np.cumsum(S_inp**2) / np.sum(S_inp**2)
    rank_90_inp = int(np.searchsorted(cumvar_inp, 0.90) + 1)
    rank_99_inp = int(np.searchsorted(cumvar_inp, 0.99) + 1)

    logger.info(f'activity rank(90%)={rank_90_act}  rank(99%)={rank_99_act}')
    logger.info(f'visual input rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}')

    # print('plot kinograph ...')
    # plot_kinograph(
    #     activity=activity_full.T,
    #     stimulus=x_ts.stimulus[:, :sim.n_input_neurons].numpy().T,
    #     output_path=graphs_data_path(config.dataset, 'kinograph.png'),
    #     rank_90_act=rank_90_act,
    #     rank_99_act=rank_99_act,
    #     rank_90_inp=rank_90_inp,
    #     rank_99_inp=rank_99_inp,
    #     zoom_size=200,
    #     style=fig_style,
    # )

    logger.info('plot activity traces ...')
    trace_name = f'activity_traces_mask_{int(sim.ablation_ratio*100)}.png' if sim.ablation_ratio > 0 else 'activity_traces.png'
    trace_indices = plot_activity_traces(
        activity=activity_full.T,
        output_path=graphs_data_path(config.dataset, trace_name),
        n_traces=100,
        max_frames=10000,
        n_input_neurons=sim.n_input_neurons,
        style=fig_style,
        dpi=300,
    )

    # Plot noisy activity traces using the same neurons + compute SNR
    snr_stats = None
    if sim.measurement_noise_level > 0:
        logger.debug('plot noisy activity traces ...')
        noise_data = x_ts.noise.numpy() if x_ts.noise is not None else None
        if noise_data is not None:
            noisy_activity = activity_full + noise_data  # (T, N)
            plot_activity_traces(
                activity=noisy_activity.T,
                output_path=graphs_data_path(config.dataset, 'activity_traces_noisy.png'),
                n_traces=100,
                max_frames=10000,
                n_input_neurons=sim.n_input_neurons,
                style=fig_style,
                neuron_indices=trace_indices,
                dpi=300,
                title='noisy voltage traces (measurement noise)',
            )

            # --- SNR analysis (per neuron) ---
            # Voltage SNR: std(clean_voltage) / std(measurement_noise) per neuron
            signal_std = np.std(activity_full, axis=0)  # (N,)
            noise_std = np.std(noise_data, axis=0)       # (N,)
            voltage_snr = node_params.Where(noise_std > 0, signal_std / noise_std, np.inf)
            voltage_snr_finite = voltage_snr[np.isfinite(voltage_snr)]

            # Derivative SNR: std(clean_derivative) / std(derivative_noise) per neuron
            # derivative noise = (noise[t+1] - noise[t]) / dt
            deriv_noise = np.diff(noise_data, axis=0) / sim.delta_t  # (T-1, N)
            deriv_noise_std = np.std(deriv_noise, axis=0)             # (N,)
            y_clean = load_raw_array(graphs_data_path(config.dataset, 'y_list_train'))  # (T, N, 1)
            deriv_signal_std = np.std(y_clean[:, :, 0], axis=0)  # (N,)
            deriv_snr = node_params.Where(deriv_noise_std > 0, deriv_signal_std / deriv_noise_std, np.inf)
            deriv_snr_finite = deriv_snr[np.isfinite(deriv_snr)]

            deriv_noise_std_theoretical = sim.measurement_noise_level * np.sqrt(2) / sim.delta_t
            deriv_noise_std_empirical = np.mean(deriv_noise_std)

            snr_stats = {
                'voltage_snr_mean': np.mean(voltage_snr_finite),
                'voltage_snr_median': np.median(voltage_snr_finite),
                'voltage_snr_min': np.min(voltage_snr_finite),
                'voltage_snr_max': np.max(voltage_snr_finite),
                'derivative_snr_mean': np.mean(deriv_snr_finite),
                'derivative_snr_median': np.median(deriv_snr_finite),
                'derivative_snr_min': np.min(deriv_snr_finite),
                'derivative_snr_max': np.max(deriv_snr_finite),
                'derivative_noise_std_theoretical': deriv_noise_std_theoretical,
                'derivative_noise_std_empirical': deriv_noise_std_empirical,
            }

            logger.info('--- Measurement noise SNR analysis ---')
            logger.info('  voltage SNR (std_signal / std_noise) per neuron:')
            logger.info(f'    mean: {snr_stats["voltage_snr_mean"]:.2f}  '
                        f'median: {snr_stats["voltage_snr_median"]:.2f}  '
                        f'min: {snr_stats["voltage_snr_min"]:.2f}  '
                        f'max: {snr_stats["voltage_snr_max"]:.2f}')
            logger.info('  derivative SNR (std_dy/dt / std_noise_dy/dt) per neuron:')
            logger.info(f'    mean: {snr_stats["derivative_snr_mean"]:.2f}  '
                        f'median: {snr_stats["derivative_snr_median"]:.2f}  '
                        f'min: {snr_stats["derivative_snr_min"]:.2f}  '
                        f'max: {snr_stats["derivative_snr_max"]:.2f}')
            logger.info(f'  derivative noise std (theoretical): '
                        f'{snr_stats["derivative_noise_std_theoretical"]:.2f}')
            logger.info(f'  derivative noise std (empirical mean): '
                        f'{snr_stats["derivative_noise_std_empirical"]:.2f}')
            logger.info('--------------------------------------')

    # SVD analysis (4-panel plot)
    logger.info('svd analysis ...')
    from flyvis_gnn.models.utils import analyze_data_svd
    folder = graphs_data_path(config.dataset)
    svd_results = analyze_data_svd(x_ts, folder, config=config, is_flyvis=True,
                                   save_in_subfolder=False, logger=logger)

    # Save ranks to log file
    gen_log_path = graphs_data_path(config.dataset, 'generation_log.txt')
    with open(gen_log_path, 'w') as log_f:
        log_f.write(f'dataset: {config.dataset}\n')
        log_f.write(f'n_neurons: {n_neurons}\n')
        log_f.write(f'n_input_neurons: {sim.n_input_neurons}\n')
        log_f.write(f'n_frames_train: {n_frames_train}\n')
        log_f.write(f'n_frames_test: {n_frames_test}\n')
        log_f.write(f'n_sequences_train: {len(train_sequences)}\n')
        log_f.write(f'n_sequences_test: {len(test_sequences)}\n')
        log_f.write(f'n_train_videos: {n_train_vids}\n')
        log_f.write(f'n_test_videos: {len(test_video_set)}\n')
        log_f.write(f'train_videos: {train_video_names}\n')
        log_f.write(f'test_videos: {test_video_names}\n')
        log_f.write(f'visual_input_type: {sim.visual_input_type}\n')
        log_f.write(f'noise_model_level: {sim.noise_model_level}\n')
        log_f.write(f'measurement_noise_level: {sim.measurement_noise_level}\n')
        log_f.write(f'model_id: {sim.model_id}\n')
        log_f.write(f'ensemble_id: {sim.ensemble_id}\n')
        log_f.write('\n')
        log_f.write(f'activity_rank_90: {rank_90_act}\n')
        log_f.write(f'activity_rank_99: {rank_99_act}\n')
        log_f.write(f'input_rank_90: {rank_90_inp}\n')
        log_f.write(f'input_rank_99: {rank_99_inp}\n')
        if svd_results.get('activity'):
            log_f.write(f'svd_activity_rank_90: {svd_results["activity"]["rank_90"]}\n')
            log_f.write(f'svd_activity_rank_99: {svd_results["activity"]["rank_99"]}\n')
        if svd_results.get('visual_stimuli'):
            log_f.write(f'svd_visual_rank_90: {svd_results["visual_stimuli"]["rank_90"]}\n')
            log_f.write(f'svd_visual_rank_99: {svd_results["visual_stimuli"]["rank_99"]}\n')
        if snr_stats is not None:
            log_f.write('\n')
            for key, val in snr_stats.items():
                log_f.write(f'{key}: {val:.2f}\n')
    logger.info(f'generation log saved to {gen_log_path}')

    if not visualize:
        return

    # Neuron type index to name mapping (CamelCase for legacy plot_neuron_activity_analysis)
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
        18: 'Mi15', 19: 'Mi2', 20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
        27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
        36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1',
        44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
        52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14',
        59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
    }

    activity = x_ts.voltage.to(device).t()  # (n_neurons, n_frames)
    type_list = x.neuron_type.unsqueeze(-1).to(device)

    target_type_name_list = ['R1', 'R7', 'C2', 'Mi11', 'Tm1', 'Tm4', 'Tm30']
    from GNN_PlotFigure import plot_neuron_activity_analysis
    plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, sim.delta_t, graphs_data_path(config.dataset) + '/')

    logger.info('plot figure activity ...')
    plot_selected_neuron_traces(
        activity=to_numpy(activity),
        type_list=to_numpy(type_list.squeeze()),
        output_path=graphs_data_path(config.dataset, 'activity.png'),
        style=fig_style,
    )

    if visualize & (run == run_vizualized):
        logger.info('generating lossless video ...')

        output_name = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'no_id'
        src = graphs_data_path(config.dataset, "Fig", "Fig_0_000000.png")
        dst = graphs_data_path(config.dataset, f"input_{output_name}.png")
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=graphs_data_path(config.dataset), run=run,
                                      output_name=output_name,framerate=20)

        files = glob.glob(graphs_data_path(config.dataset, 'Fig', '*'))
        for f in files:
            os.remove(f)



