"""FlyVis-GNN — Local LLM Exploration Loop.

Lightweight version of GNN_LLM.py for running agentic hyper-parameter
optimization locally (no SSH, no cluster submission).

Pipeline structure:
  setup → batch_0 → loop { load → train(local) → artifacts → UCB → analysis → finalize }

Usage:
  python GNN_Agentic.py -o generate_train_test_plot_Claude flyvis_noise_005 iterations=168
  python GNN_Agentic.py -o generate_train_test_plot_Claude flyvis_noise_005 --resume
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import warnings

from flyvis_gnn.LLM import (
    setup_exploration,
    init_slot_configs,
    init_shared_files,
    make_batch_info,
    run_batch_0,
    load_configs_and_seeds,
    run_local_pipeline,
    save_artifacts,
    update_ucb_scores,
    run_claude_analysis,
    finalize_batch,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(description="FlyVis-GNN — Local LLM Exploration")
    parser.add_argument("-o", "--option", nargs="+", help="option that takes multiple values")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="start from iteration 1 (ignore auto-resume)")
    parser.add_argument("--resume", action="store_true",
                        help="auto-resume from last completed batch")
    args = parser.parse_args()
    # Force local mode (no cluster)
    args.cluster = False
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Setup ---
    state = setup_exploration(args, root_dir)
    # Disable interactive code sessions (not needed for local exploration)
    state.interaction_code = False
    init_slot_configs(state, is_resume=args.resume)
    init_shared_files(state, is_resume=args.resume)

    # --- Batch 0: initialize config variations (fresh start only) ---
    if state.start_iteration == 1 and not args.resume:
        run_batch_0(state)

    # --- Main batch loop ---
    for batch_start in range(state.start_iteration, state.n_iterations + 1, state.n_parallel):
        batch = make_batch_info(state, batch_start)

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}  (block {batch.block_number})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # Load configs + force seeds
        load_configs_and_seeds(state, batch)

        # Training (local only)
        if "train" in state.task:
            run_local_pipeline(state, batch)
        else:
            # No training — mark all slots as successful
            for slot in range(batch.n_slots):
                batch.job_results[slot] = True

        # Save exploration artifacts
        save_artifacts(state, batch)

        # Compute UCB scores
        update_ucb_scores(state, batch)

        # Claude analysis + next mutations
        run_claude_analysis(state, batch)

        # Finalize: tree viz, protocol/memory snapshots
        finalize_batch(state, batch)
