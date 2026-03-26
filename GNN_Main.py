import sys
import os

# Ensure src/ is on the path so flyvis_gnn is always importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)


from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.models.graph_trainer import data_train, data_test, data_train_INR
from flyvis_gnn.utils import set_device, add_pre_folder, log_path, config_path

# Optional imports (not available in flyvis-gnn spinoff)
try:
    from flyvis_gnn.models.NGP_trainer import data_train_NGP
except ImportError:
    data_train_NGP = None
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="flyvis_gnn")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )

    print()
    device = []
    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    CONFIG_LISTS = {
        'known_ode': [
            'flyvis_noise_free_known_ode',
            'flyvis_noise_005_known_ode',
            'flyvis_noise_05_known_ode',
            'flyvis_noise_005_INR_known_ode',
        ],
    }

    if args.option is not None:
        task = args.option[0]
        config_name = args.option[1]
        if config_name in CONFIG_LISTS:
            config_list = CONFIG_LISTS[config_name]
            best_model = None
            test_config_name = None
        else:
            config_list = [config_name]
            if len(args.option) > 2:
                best_model = args.option[2]
            else:
                best_model = None
            if len(args.option) > 3:
                test_config_name = args.option[3]
            else:
                test_config_name = None
    else:
        best_model = ''
        task = task = 'traimn'
        config_list = ['flyvis_noise_005']
        test_config_name = None

    for config_file_ in config_list:
        print(" ")
        config_file, pre_folder = add_pre_folder(config_file_)

        # load config
        config = NeuralGraphConfig.from_yaml(config_path(f"{config_file}.yaml"))
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_

        if device == []:
            device = set_device(config.training.device)

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=False,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                save=True,
                step=100,
            )

        if 'train_NGP' in task:
            # use new modular NGP trainer pipeline
            data_train_NGP(config=config, device=device)

        elif 'train_INR' in task:
            # train INR (SIREN/NGP) on a field from x_list_train
            field_name = args.option[2] if len(args.option) > 2 else 'stimulus'
            data_train_INR(config=config, device=device, total_steps=100000,
                           field_name=field_name, n_training_frames=1000,
                           inr_type='siren_txy')

        elif "train" in task:
            data_train(
                config=config,
                erase=True,
                best_model=best_model,
                style='color',
                device=device,
            )

        if "test" in task:
            config.simulation.noise_model_level = 0.0

            # Optional: load a second config for cross-dataset test data
            test_config = None
            if test_config_name:
                tc_file, tc_pre = add_pre_folder(test_config_name)
                test_config = NeuralGraphConfig.from_yaml(f"{config_root}/{tc_file}.yaml")
                test_config.dataset = tc_pre + test_config.dataset
                test_config.config_file = tc_pre + test_config_name
                print(f'cross-dataset test: model from {config.dataset}, test data from {test_config.dataset}')

            data_test(
                config=config,
                visualize=True,
                style="color name continuous_slice",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",   # test_ablation_50
                sample_embedding=False,
                step=10,
                n_rollout_frames=250,
                device=device,
                particle_of_interest=0,
                new_params=None,
                rollout_without_noise=False,
                test_config=test_config,
            )

        if 'plot' in task:
            folder_name = log_path(pre_folder, 'tmp_results') + '/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True)


# python GNN_Main.py -o test flyvis_noise_005 best flyvis_noise_free