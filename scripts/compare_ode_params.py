"""Compare ode_params between two datasets.

Usage:
    python scripts/compare_ode_params.py flyvis_noise_05 flyvis_noise_05_03
"""
import sys
import torch
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flyvis_gnn.utils import graphs_data_path
from flyvis_gnn.generators.ode_params import FlyVisODEParams


def describe(name, tensor):
    if tensor is None:
        print(f"  {name}: None")
        return
    print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}")


def compare(dataset_new, dataset_old):
    print(f"\n=== Comparing '{dataset_new}' (new) vs '{dataset_old}' (old) ===\n")

    folder_new = graphs_data_path(dataset_new)
    folder_old = graphs_data_path(dataset_old)

    # Load new
    new = FlyVisODEParams.load(folder_new, device='cpu')
    print(f"NEW ({dataset_new}):")
    describe("W", new.W)
    describe("edge_index", new.edge_index)
    describe("tau_i", new.tau_i)
    describe("V_i_rest", new.V_i_rest)

    # Load old
    old = FlyVisODEParams.load(folder_old, device='cpu')
    print(f"\nOLD ({dataset_old}):")
    describe("W", old.W)
    describe("edge_index", old.edge_index)
    describe("tau_i", old.tau_i)
    describe("V_i_rest", old.V_i_rest)

    # Direct comparison
    print("\n=== COMPARISON ===")

    # W
    if new.W is not None and old.W is not None:
        if new.W.shape == old.W.shape:
            diff = (new.W - old.W).abs()
            print(f"W: shapes match {tuple(new.W.shape)}, "
                  f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
        else:
            print(f"W: SHAPE MISMATCH — new={tuple(new.W.shape)}, old={tuple(old.W.shape)}")
            # Try transposed comparison
            if new.W.shape == old.W.T.shape:
                diff = (new.W - old.W.T).abs()
                print(f"  -> W vs W.T match: max_diff={diff.max().item():.6f}")
            if new.W.flatten().shape == old.W.flatten().shape:
                diff = (new.W.flatten() - old.W.flatten()).abs()
                print(f"  -> flattened max_diff={diff.max().item():.6f}")

    # edge_index
    if new.edge_index is not None and old.edge_index is not None:
        if new.edge_index.shape == old.edge_index.shape:
            diff = (new.edge_index - old.edge_index).abs()
            print(f"edge_index: shapes match {tuple(new.edge_index.shape)}, "
                  f"max_diff={diff.max().item()}")
        else:
            print(f"edge_index: SHAPE MISMATCH — new={tuple(new.edge_index.shape)}, "
                  f"old={tuple(old.edge_index.shape)}")
            # Check if it's transposed (2, E) vs (E, 2)
            if new.edge_index.shape == old.edge_index.T.shape:
                diff = (new.edge_index - old.edge_index.T).abs()
                print(f"  -> edge_index vs edge_index.T: max_diff={diff.max().item()}")

    # tau_i
    if new.tau_i is not None and old.tau_i is not None:
        if new.tau_i.shape == old.tau_i.shape:
            diff = (new.tau_i - old.tau_i).abs()
            print(f"tau_i: shapes match {tuple(new.tau_i.shape)}, "
                  f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
        else:
            print(f"tau_i: SHAPE MISMATCH — new={tuple(new.tau_i.shape)}, old={tuple(old.tau_i.shape)}")

    # V_i_rest
    if new.V_i_rest is not None and old.V_i_rest is not None:
        if new.V_i_rest.shape == old.V_i_rest.shape:
            diff = (new.V_i_rest - old.V_i_rest).abs()
            print(f"V_i_rest: shapes match {tuple(new.V_i_rest.shape)}, "
                  f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
        else:
            print(f"V_i_rest: SHAPE MISMATCH — new={tuple(new.V_i_rest.shape)}, old={tuple(old.V_i_rest.shape)}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_ode_params.py <new_dataset> <old_dataset>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
