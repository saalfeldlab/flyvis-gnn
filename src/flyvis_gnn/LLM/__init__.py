"""LLM exploration pipeline for flyvis-gnn.

Provides the infrastructure for Claude-driven hyperparameter exploration
with optional interactive code modification sessions at block boundaries.
"""

from .pipeline import (
    finalize_batch,
    generate_data_locally,
    init_shared_files,
    init_slot_configs,
    load_configs_and_seeds,
    make_batch_info,
    run_batch_0,
    run_claude_analysis,
    run_cluster_training,
    run_code_session,
    run_local_pipeline,
    run_local_test_plot,
    save_artifacts,
    setup_exploration,
    update_ucb_scores,
)
from .state import BatchInfo, ExplorationState

__all__ = [
    'ExplorationState',
    'BatchInfo',
    'setup_exploration',
    'init_slot_configs',
    'init_shared_files',
    'make_batch_info',
    'run_batch_0',
    'run_code_session',
    'load_configs_and_seeds',
    'generate_data_locally',
    'run_cluster_training',
    'run_local_test_plot',
    'run_local_pipeline',
    'save_artifacts',
    'update_ucb_scores',
    'run_claude_analysis',
    'finalize_batch',
]
