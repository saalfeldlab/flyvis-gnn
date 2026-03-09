"""Standalone training script for cluster jobs.

Called by the LLM exploration pipeline (cluster.py) to run training only
on a cluster node. Data generation and test/plot are handled locally.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')
import argparse
import traceback

if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='flyvis training subprocess')
    parser.add_argument('--config', required=True, help='path to YAML config')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log_file', default=None, help='path for analysis log')
    parser.add_argument('--config_file', default=None, help='config_file field (e.g. fly/flyvis_noise_05_Claude_00)')
    parser.add_argument('--error_log', default=None, help='path for error details')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--exploration_dir', default=None)
    parser.add_argument('--iteration', type=int, default=None)
    parser.add_argument('--slot', type=int, default=None)
    args = parser.parse_args()

    try:
        config = NeuralGraphConfig.from_yaml(args.config)
        if args.config_file:
            config.config_file = args.config_file

        log_file = open(args.log_file, 'w') if args.log_file else None
        try:
            data_train(
                config=config,
                erase=args.erase,
                device=args.device,
                log_file=log_file,
            )
        finally:
            if log_file:
                log_file.close()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if args.error_log:
            with open(args.error_log, 'a') as f:
                f.write(f"\n--- iteration {args.iteration} slot {args.slot} ---\n")
                f.write(tb)
        sys.exit(1)
