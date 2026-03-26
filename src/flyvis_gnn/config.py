from enum import Enum
from typing import Annotated, Optional


# Python 3.10 compatibility (StrEnum added in 3.11)
class StrEnum(str, Enum):
    pass
import yaml
from pydantic import BaseModel, ConfigDict, Field

# StrEnum types for config fields

class Boundary(StrEnum):
    PERIODIC = "periodic"
    NO = "no"
    PERIODIC_SPECIAL = "periodic_special"
    WALL = "wall"

class ExternalInputType(StrEnum):
    NONE = "none"
    SIGNAL = "signal"
    VISUAL = "visual"
    MODULATION = "modulation"

class ExternalInputMode(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    NONE = "none"

class SignalInputType(StrEnum):
    OSCILLATORY = "oscillatory"
    TRIGGERED = "triggered"

class CalciumType(StrEnum):
    NONE = "none"
    LEAKY = "leaky"
    MULTI_COMPARTMENT = "multi-compartment"
    SATURATION = "saturation"

class CalciumActivation(StrEnum):
    SOFTPLUS = "softplus"
    RELU = "relu"
    IDENTITY = "identity"
    TANH = "tanh"

class Prediction(StrEnum):
    FIRST_DERIVATIVE = "first_derivative"
    SECOND_DERIVATIVE = "2nd_derivative"
    NEXT_ACTIVITY = "next_activity"

class Integration(StrEnum):
    EULER = "Euler"
    RUNGE_KUTTA = "Runge-Kutta"

class UpdateType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"
    PRE_MLP = "pre_mlp"
    TWO_STEPS = "2steps"
    NONE = "none"
    NO_POS = "no_pos"
    GENERIC = "generic"
    EXCITATION = "excitation"
    GENERIC_EXCITATION = "generic_excitation"
    EMBEDDING_MLP = "embedding_MLP"
    TEST_FIELD = "test_field"

class MLPActivation(StrEnum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    SOFT_RELU = "soft_relu"
    NONE = "none"

class INRType(StrEnum):
    SIREN_T = "siren_t"
    SIREN_TXY = "siren_txy"
    SIREN_ID = "siren_id"
    SIREN_X = "siren_x"
    NGP = "ngp"
    LOWRANK = "lowrank"

class DenoiserType(StrEnum):
    NONE = "none"
    WINDOW = "window"
    LSTM = "LSTM"
    GAUSSIAN_FILTER = "Gaussian_filter"
    WAVELET = "wavelet"

class GhostMethod(StrEnum):
    NONE = "none"
    TENSOR = "tensor"
    MLP = "MLP"

class Sparsity(StrEnum):
    NONE = "none"
    REPLACE_EMBEDDING = "replace_embedding"
    REPLACE_EMBEDDING_FUNCTION = "replace_embedding_function"
    REPLACE_STATE = "replace_state"
    REPLACE_TRACK = "replace_track"

class ClusterMethod(StrEnum):
    KMEANS = "kmeans"
    KMEANS_AUTO_PLOT = "kmeans_auto_plot"
    KMEANS_AUTO_EMBEDDING = "kmeans_auto_embedding"
    DISTANCE_PLOT = "distance_plot"
    DISTANCE_EMBEDDING = "distance_embedding"
    DISTANCE_BOTH = "distance_both"
    INCONSISTENT_PLOT = "inconsistent_plot"
    INCONSISTENT_EMBEDDING = "inconsistent_embedding"
    NONE = "none"

class ClusterConnectivity(StrEnum):
    SINGLE = "single"
    AVERAGE = "average"

class OdeMethod(StrEnum):
    DOPRI5 = "dopri5"
    RK4 = "rk4"
    EULER = "euler"
    MIDPOINT = "midpoint"
    HEUN3 = "heun3"

class WInitMode(StrEnum):
    RANDN = "randn"
    RANDN_SCALED = "randn_scaled"
    ZEROS = "zeros"

class GPhiMode(StrEnum):
    MLP = "mlp"
    TANH = "tanh"
    IDENTITY = "identity"

class WOptimizerType(StrEnum):
    ADAM = "adam"
    SGD = "sgd"

class UmapClusterMethod(StrEnum):
    NONE = "none"
    DBSCAN = "dbscan"
    GMM = "gmm"

class LabelStyle(StrEnum):
    MLP = "MLP"
    GREEK = "greek"


# Sub-config schemas for NeuralGraph


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dimension: int = 2
    n_frames: int = 1000  # number of simulation time steps; 0 = use each source frame exactly once (no reuse)
    start_frame: int = 0
    seed: int = 42

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Boundary = Boundary.PERIODIC
    min_radius: float = 0.0
    max_radius: float = 0.1

    n_neurons: int = 1000
    n_neuron_types: int = 5
    n_input_neurons: int = 0
    n_edges: int = 0
    max_edges: float = 1.0e6
    n_extra_null_edges: int = 0
    null_edges_mode: str = "per_column"  # "random" or "per_column" (per pre-synaptic neuron)
    edge_removal_ratio: float = 0.0  # fraction of edges to remove before saving (0.0-1.0)
    edge_removal_mode: str = "random"  # "random" or "per_column"
    edge_removal_seed: int = 42      # RNG seed for reproducible removal
    ablation_ratio: float = 0.0   # fraction of edges to ablate (0.0-1.0)
    ablation_seed: int = 42       # RNG seed for reproducible ablation

    baseline_value: float = -999.0
    shuffle_neuron_types: bool = False

    noise_visual_input: float = 0.0
    only_noise_visual_input: float = 0.0
    visual_input_type: str = ""  # for flyvis experiments
    datavis_roots: list[str] = []  # list of dataset roots (each contains JPEGImages/480p/); empty list uses default get_datavis_root_dir()
    skip_short_videos: bool = True  # skip videos with fewer frames than chunk size (n_frames in video_config)
    max_train_sequences: int = 0  # limit train sequences (0 = use all); reduces generation time proportionally
    blank_freq: int = 2  # Frequency of blank frames in visual input
    simulation_initial_state: bool = False


    # external input configuration
    external_input_type: ExternalInputType = ExternalInputType.NONE
    external_input_mode: ExternalInputMode = ExternalInputMode.NONE
    permutation: bool = False  # whether to apply random permutation to external input

    # signal input parameters (external_input_type == "signal")
    signal_input_type: SignalInputType = SignalInputType.OSCILLATORY
    oscillation_max_amplitude: float = 1.0
    oscillation_frequency: float = 5.0

    # triggered oscillation parameters (signal_input_type == "triggered")
    triggered_n_impulses: int = 5  # number of impulse events
    triggered_n_input_neurons: int = 10  # number of neurons receiving impulse input per event
    triggered_impulse_strength: float = 5.0  # base strength of impulse (will vary randomly)
    triggered_min_start_frame: int = 50  # minimum frame for first trigger
    triggered_max_start_frame: int = 150  # maximum frame for first trigger (ignored if n_impulses > 1)
    triggered_duration_frames: int = 200  # duration of oscillation response per impulse
    triggered_amplitude_range: list[float] = [0.5, 2.0]  # min/max amplitude multiplier
    triggered_frequency_range: list[float] = [0.5, 2.0]  # min/max frequency multiplier

    tile_contrast: float = 0.2
    tile_corr_strength: float = 0.0   # correlation knob for tile_mseq / tile_blue_noise
    tile_flip_prob: float = 0.05      # per-frame random flip probability
    tile_seed: int = 42

    n_nodes: Optional[int] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"

    adjacency_matrix: str = ""
    short_term_plasticity_mode: str = "depression"

    # AdEx spiking model parameters
    adex_dt: float = 0.2              # ms — integration timestep for AdEx (0.2ms default from Zerlaut)
    adex_stim_scale: float = 1.0      # pA per unit stimulus — converts visual input to current
    adex_I_bias: float = 0.0          # pA — constant bias current injected into all neurons

    # Hodgkin-Huxley model parameters
    hh_substeps: int = 50             # number of Euler substeps per stimulus frame
    hh_stim_scale: float = 50.0       # uA/cm^2 per unit stimulus
    hh_I_bias: float = 3.0            # uA/cm^2 — tonic drive (subthreshold)
    hh_w_scale: float = 2.0           # global W multiplier (connectome weights calibrated for graded model)

    connectivity_file: str = ""
    connectivity_init: list[float] = [-1]
    connectivity_filling_factor: float = 1
    connectivity_type: str = "none"  # none, Lorentz, Gaussian, uniform, chaotic, ring attractor, low_rank, successor, null, Lorentz_structured_X_Y
    connectivity_rank: int = 1
    connectivity_parameter: float = 1.0

    Dale_law: bool = False
    Dale_law_factor: float = 0.5  # fraction of excitatory (positive) columns, rest are inhibitory

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    params: list[list[float]]
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    calcium_type: CalciumType = CalciumType.NONE
    calcium_activation: CalciumActivation = CalciumActivation.SOFTPLUS
    calcium_tau: float = 0.5  # decay time constant (same units as delta_t)
    calcium_alpha: float = 1.0  # scale factor to convert [Ca] to fluorescence
    calcium_beta: float = 0.0  # baseline offset for fluorescence
    calcium_initial: float = 0.0  # initial calcium concentration
    calcium_noise_level: float = 0.0  # optional Gaussian noise added to [Ca] updates
    noise_model_level: float = 0.0  # process noise added during dynamics simulation
    measurement_noise_level: float = 0.0  # observation noise saved separately in noise.zarr
    derivative_smoothing_window: int = 1  # temporal smoothing window for noisy derivatives (1 = no smoothing)
    calcium_saturation_kd: float = 1.0  # for nonlinear saturation models
    calcium_num_compartments: int = 1
    calcium_down_sample: int = 1  # down-sample [Ca] time series by this factor
    save_calcium: bool = False  # save calcium and fluorescence to zarr (large files)

    pos_init: str = "uniform"
    dpos_init: float = 0



class ClaudeConfig(BaseModel):
    """Configuration for Claude-driven exploration experiments."""
    model_config = ConfigDict(extra="ignore")

    n_epochs: int = 1  # number of epochs per iteration
    data_augmentation_loop: int = 100  # data augmentation loop count
    n_iter_block: int = 24  # number of iterations per simulation block
    ucb_c: float = 1.414  # UCB exploration constant: UCB(k) = R²_k + c * sqrt(ln(N) / n_k)
    n_parallel: int = 4  # number of parallel config slots per batch
    generate_data: bool = False  # generate new simulation data before each training iteration
    training_time_target_min: int = 60  # target training time per iteration in minutes (for LLM guidance)


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    particle_model_name: str = ""
    cell_model_name: str = ""
    mesh_model_name: str = ""
    signal_model_name: str = ""
    prediction: Prediction = Prediction.SECOND_DERIVATIVE
    integration: Integration = Integration.EULER

    aggr_type: str
    embedding_dim: int = 2

    field_type: str = ""
    field_grid: Optional[str] = ""

    input_size: int = 1
    output_size: int = 1
    hidden_dim: int = 1
    n_layers: int = 1

    input_size_2: int = 1
    output_size_2: int = 1
    hidden_dim_2: int = 1
    n_layers_2: int = 1


    input_size_decoder: int = 1
    output_size_decoder: int = 1
    hidden_dim_decoder: int = 1
    n_layers_decoder: int = 1

    input_size_encoder: int = 1
    output_size_encoder: int = 1
    hidden_dim_encoder: int = 1
    n_layers_encoder: int = 1

    g_phi_positive: bool = False

    update_type: UpdateType = UpdateType.NONE

    MLP_activation: MLPActivation = MLPActivation.RELU


    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1

    kernel_type: str = "mlp"

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1
    outermost_linear_nnr: bool = True
    omega: float = 80.0

    input_size_nnr_f: int = 3
    n_layers_nnr_f: int = 5
    hidden_dim_nnr_f: int = 128
    output_size_nnr_f: int = 1
    outermost_linear_nnr_f: bool = True
    omega_f: float = 80.0
    omega_f_learning: bool = False  # make omega learnable during training

    nnr_f_xy_period: float = 1.0
    nnr_f_T_period: float = 1.0

    # INR type for external input learning
    # siren_t: input=t, output=n_neurons (current implementation, works for n_neurons < 100)
    # siren_id: input=(t, id), output=1 (scales better for large n_neurons)
    # siren_x: input=(t, x, y), output=1 (uses neuron positions)
    # ngp: instantNGP hash encoding
    # lowrank: low-rank matrix factorization U @ V (not a neural network)
    inr_type: INRType = INRType.SIREN_T

    # LowRank factorization parameters
    lowrank_rank: int = 64  # rank of the factorization (params = rank * (n_frames + n_neurons))
    lowrank_svd_init: bool = True  # initialize with SVD of the data

    # InstantNGP (hash encoding) parameters
    ngp_n_levels: int = 24
    ngp_n_features_per_level: int = 2
    ngp_log2_hashmap_size: int = 22
    ngp_base_resolution: int = 16
    ngp_per_level_scale: float = 1.4
    ngp_n_neurons: int = 128
    ngp_n_hidden_layers: int = 4

    input_size_modulation: int = 2
    n_layers_modulation: int = 3
    hidden_dim_modulation: int = 64
    output_size_modulation: int = 1

    input_size_excitation: int = 3
    n_layers_excitation: int = 5
    hidden_dim_excitation: int = 128

    excitation_dim: int = 1

    latent_dim: int = 64
    latent_update_steps: int = 50
    stochastic_latent: bool = True
    latent_init_std: float = 1.0  # only used if you later add 'init from noise' modes

    # encoder sizes (x -> [mu, logvar])
    input_size_encoder: int = 1      # set to n_neurons in your YAML
    n_layers_encoder: int = 3
    hidden_dim_encoder: int = 256
    latent_n_layers_update: int = 2
    latent_hidden_dim_update: int = 64
    output_size_decoder: int = 1      # set to n_neurons in your YAML
    n_layers_decoder: int = 3
    hidden_dim_decoder:  int = 256


class ZarrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_fluo: str = ""
    store_seg: str = ""

    axis: int = 0
    frame: int = 0
    contrast: str = "1,99.9"
    rendering: str = "1,99.9"
    dz_um: float = 4
    dy_um: float = 0.406
    dx_um: float = 0.406
    labels_opacity: float = 0.7
    show_boundaries: bool = False


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str = "tab10"
    arrow_length: int = 10
    marker_size: int = 100
    xlim: list[float] = [-0.1, 0.1]
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]
    speedlim: list[float] = [0, 1]
    pic_folder: str = "none"
    pic_format: str = "jpg"
    pic_size: list[int] = [1000, 1100]
    data_embedding: int = 1
    plot_batch_size: int = 1000
    label_style: LabelStyle = LabelStyle.MLP  # MLP for MLP_0/MLP_1 labels; greek for phi/f labels

    # MLP plot axis limits
    mlp0_xlim: list[float] = [-5, 5]
    mlp0_ylim: list[float] = [-8, 8]
    mlp1_xlim: list[float] = [-5, 5]
    mlp1_ylim: list[float] = [-1.1, 1.1]

    # MLP normalization settings
    norm_method: str = "median"
    norm_x_start: float | None = None  # None = auto (0.85 * xnorm * 4 for training, 0.8 * xnorm for best)
    norm_x_stop: float | None = None   # None = auto (xnorm * 4 for training, xnorm for best)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"

    n_epochs: int = 20
    n_epochs_init: int = 99999  # DEPRECATED: no longer used by regularizer
    epoch_reset: int = -1
    epoch_reset_freq: int = 99999
    batch_size: int = 1
    inr_batch_size: int = 8
    n_training_frames: int = 0  # 0 = use all frames; >0 = crop centered window
    batch_ratio: float = 1
    small_init_batch_size: bool = True
    embedding_step: int = 1000
    shared_embedding: bool = False
    embedding_trial: bool = False
    remove_self: bool = True

    pretrained_model: str = ""
    pre_trained_W: str = ""

    multi_connectivity: bool = False
    with_connectivity_mask: bool = False
    has_missing_activity: bool = False

    epoch_distance_replace: int = 20
    warm_up_length: int = 10
    sequence_length: int = 32

    denoiser: bool = False
    denoiser_type: DenoiserType = DenoiserType.NONE
    denoiser_param: float = 1.0

    training_selected_neurons: bool = False
    selected_neuron_ids: list[int] = [1]

    time_window: int = 0

    n_runs: int = 2
    seed: int = 42
    clamp: float = 0
    pred_limit: float = 1.0e10

    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: GhostMethod = GhostMethod.NONE
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Sparsity = Sparsity.NONE
    fix_cluster_embedding: bool = False
    cluster_method: ClusterMethod = ClusterMethod.DISTANCE_PLOT
    cluster_distance_threshold: float = 0.1
    cluster_connectivity: ClusterConnectivity = ClusterConnectivity.SINGLE

    umap_cluster_method: UmapClusterMethod = UmapClusterMethod.NONE
    umap_cluster_freq: int = 1
    umap_cluster_n_neighbors: int = 50
    umap_cluster_min_dist: float = 0.1
    umap_cluster_eps: float = 0.1
    umap_cluster_gmm_n: int = 50
    umap_cluster_fix_embedding: bool = False
    umap_cluster_fix_embedding_ratio: float = 0.0
    umap_cluster_reinit_mlps: bool = False
    umap_cluster_relearn_epochs: int = 0

    Ising_filter: str = "none"

    init_training_single_type: bool = False
    training_single_type: bool = False

    low_rank_factorization: bool = False
    low_rank: int = 20

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_update_start: float = 0.0
    learning_rate_modulation_start: float = 0.0001
    learning_rate_W_start: float = 0.0001

    learning_rate_end: float = 0.0005
    learning_rate_embedding_end: float = 0.0001
    learning_rate_modulation_end: float = 0.0001
    Learning_rate_W_end: float = 0.0001

    learning_rate_missing_activity: float = 0.0001
    learning_rate_NNR: float = 0.0001
    learning_rate_NNR_f_start: float = 0.0
    learning_rate_NNR_f: float = 0.0001
    learning_rate_omega_f: float = 0.0001
    coeff_omega_f_L2: float = 0.0
    training_NNR_start_epoch: int = 0

    coeff_W_L1: float = 0.0
    coeff_W_L2: float = 0.0
    coeff_W_sign: float = 0
    W_sign_temperature: float = 10.0

    # Shared annealing rate for all weight regularization (L1 and L2)
    # Formula: coeff * (1 - exp(-rate * epoch)). With rate=0.5, ramps from 0 at
    # epoch 0 to ~0.39x at epoch 1 to ~0.92x at epoch 5. Set to 0 to disable.
    regul_annealing_rate: float = 0.5

    # Regularization coefficients
    # -- f_theta (MLP0, neuron update) regularizers --
    coeff_f_theta_zero: float = 0  # Penalize f_theta(0) != 0 (enforce zero-input zero-output)
    coeff_f_theta_diff: float = 0  # Monotonicity of f_theta w.r.t. voltage (generic update_type only)
    coeff_f_theta_msg_diff: float = 0  # Monotonicity of f_theta w.r.t. aggregated message input
    coeff_f_theta_msg_sign: float = 0  # Sign consistency: f_theta output should match message sign
    coeff_func_f_theta: float = 0.0  # Penalize f_theta output at zero input
    coeff_f_theta_weight_L1: float = 0  # L1 penalty on f_theta MLP weights
    coeff_f_theta_weight_L2: float = 0  # L2 penalty on f_theta MLP weights

    # -- g_phi (MLP1, edge message) regularizers --
    coeff_g_phi_diff: float = 0  # Variance penalty on g_phi output across edges
    coeff_g_phi_norm: float = 0  # Norm penalty on g_phi edge messages
    coeff_func_g_phi: float = 0.0  # Penalize g_phi output at zero input
    coeff_g_phi_weight_L1: float = 0  # L1 penalty on g_phi MLP weights
    coeff_g_phi_weight_L2: float = 0  # L2 penalty on g_phi MLP weights

    # -- W (connectivity) regularizers --
    # coeff_W_L1, coeff_W_L2, coeff_W_sign defined above

    # -- Other regularizers --
    coeff_entropy_loss: float = 0  # Entropy penalty on predictions
    coeff_permutation: float = 100  # Permutation invariance penalty
    coeff_TV_norm: float = 0  # Total variation norm on predictions
    coeff_missing_activity: float = 0  # Penalty for missing activity patterns
    coeff_model_a: float = 0  # Regularizer on embedding a
    coeff_model_b: float = 0  # Regularizer on bias b
    coeff_lin_modulation: float = 0  # Regularizer on modulation network

    # -- f_theta linearity regularizer (unsupervised V_rest recovery) --
    coeff_f_theta_linearity: float = 0.0           # Penalize f_theta nonlinearity (0 = disabled)
    f_theta_linearity_warmup_fraction: float = 0.3  # Fraction of iterations before activation
    f_theta_linearity_rampup_iters: int = 200       # Linear ramp-up after warmup ends

    # -- f_theta centering loss (unsupervised V_rest proxy) --
    coeff_f_theta_centering: float = 0.0   # Weight of centering loss (0 = disabled)
    f_theta_centering_warmup_fraction: float = 0.3   # Fraction of iters before activation
    f_theta_centering_rampup_iters: int = 200        # Linear ramp-up after warmup

    g_phi_mode: GPhiMode = GPhiMode.MLP  # mlp=learned MLP, tanh=fixed tanh(u_j), identity=fixed u_j
    w_optimizer_type: WOptimizerType = WOptimizerType.ADAM  # adam (default) or sgd (SGD with momentum)

    # Simple training parameters (matching ParticleGraph conceptually)
    first_coeff_L1: float = 0.0  # Phase 1 weak L1 regularization
    coeff_L1: float = 0.0  # Phase 2 target L1 regularization
    coeff_diff: float = 0.0  # Monotonicity constraint on edge function

    loss_noise_level: float = 0.0

    # external input learning
    learn_external_input: bool = False

    save_all_checkpoints: bool = False  # True = save iteration-level checkpoints too

    test_dataset: str = ""  # dataset for testing; empty = same as training dataset

    data_augmentation_loop: int = 40

    recurrent_training: bool = False
    recurrent_training_start_epoch: int = 0
    recurrent_loop: int = 0
    noise_recurrent_level: float = 0.0

    neural_ODE_training: bool = False
    ode_method: OdeMethod = OdeMethod.DOPRI5
    ode_rtol: float = 1e-4
    ode_atol: float = 1e-5
    ode_adjoint: bool = True
    ode_state_clamp: float = 10.0
    ode_stab_lambda: float = 0.0
    grad_clip_W: float = 0.0
    w_init_mode: WInitMode = WInitMode.RANDN  # randn=std=1, randn_scaled=std=scale/sqrt(N), zeros
    w_init_scale: float = 1.0  # scaling factor for 'randn_scaled' mode
    coeff_W_L1_proximal: float = 0.0  # proximal L1 soft-thresholding on W after optimizer step, 0 = disabled

    alternate_training: bool = False  # two-stage training: joint warmup then V_rest focus
    alternate_joint_ratio: float = 0.4  # fraction of total iterations for joint phase (all components at full LR)
    alternate_lr_ratio: float = 0.1  # LR multiplier for W/g_phi during V_rest focus phase

    # Learning rate scheduler
    lr_scheduler: str = "none"  # 'none' | 'cosine_warm_restarts' | 'linear_warmup_cosine'
    lr_scheduler_T0: int = 1000  # restart period in iterations
    lr_scheduler_T_mult: int = 2  # period multiplier after each restart
    lr_scheduler_eta_min_ratio: float = 0.01  # min LR as fraction of base LR
    lr_scheduler_warmup_iters: int = 100  # linear warmup iterations

    time_step: int = 1
    multi_start_recurrent: bool = False
    consecutive_batch: bool = False
    recurrent_sequence: str = ""
    recurrent_parameters: list[float] = [0, 0]

    regul_matrix: bool = False
    sub_batches: int = 1
    sequence: list[str] = ["to track", "to cell"]

    MPM_trainer : str = "F"



class NeuralGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "flyvis_gnn"
    dataset: str
    data_folder_name: str = "none"
    connectome_folder_name: str = "none"
    data_folder_mesh_name: str = "none"
    config_file: str = "none"


    simulation: SimulationConfig
    graph_model: GraphModelConfig
    claude: Optional[ClaudeConfig] = None
    plotting: PlottingConfig
    training: TrainingConfig
    zarr: Optional[ZarrConfig] = None

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return NeuralGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"  # Insert path to config file
    config = NeuralGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)
