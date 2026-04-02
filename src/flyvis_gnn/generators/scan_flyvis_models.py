#!/usr/bin/env python3
# collect_flyvis_connectomes_umap_debug.py
#
# Example:
#   python collect_flyvis_connectomes_umap_debug.py \
#       --ensemble-id 0000 --start 0 --end 50 --extent 8 --out-prefix flyvis_ens0000 --debug

import argparse
import sys
import traceback
from pathlib import Path

# FlyVis
import flyvis
import numpy as np
from flyvis import NetworkView
from joblib import dump
from umap import UMAP

from flyvis_gnn.utils import setup_flyvis_model_path

setup_flyvis_model_path()


def _view_names(ensemble_id_str: str, model_id_str: str):
    e_lit, m_lit = ensemble_id_str, model_id_str
    try:
        e_pad = f"{int(ensemble_id_str):04d}"
        m_pad = f"{int(model_id_str):03d}"
    except Exception:
        e_pad, m_pad = ensemble_id_str, model_id_str
    seen, out = set(), []
    for c in (f"flow/{e_lit}/{m_lit}", f"flow/{e_pad}/{m_pad}",
              f"flow/{e_lit}/{m_pad}", f"flow/{e_pad}/{m_lit}"):
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _list_checkpoint_dirs(view_name: str):
    """Best-effort: mirror FlyVis layout seen in logs and list checkpoint dirs/files."""
    pkg_root = Path(flyvis.__file__).resolve().parent
    base = pkg_root / "data" / "results" / view_name  # e.g. .../flyvis/data/results/flow/0000/000
    ckpt_root = base / "checkpoints"
    items = []
    try:
        if ckpt_root.exists():
            items = [p.name for p in sorted(ckpt_root.iterdir())]
    except Exception:
        pass
    return base, ckpt_root, items

def _try_init_network(view_name: str, debug: bool = False):
    last_err = None
    for ckpt in (0, -1, "best"):
        try:
            if debug:
                base, ckroot, items = _list_checkpoint_dirs(view_name)
                print(f"[debug] init_network(view='{view_name}', checkpoint={ckpt} [{type(ckpt).__name__}])")
                print(f"[debug] expected dir: {base}")
                if ckroot:
                    print(f"[debug] checkpoints dir: {ckroot}  -> {items or '[]'}")
            return NetworkView(view_name).init_network(checkpoint=ckpt)
        except Exception as e:
            last_err = e
            if debug:
                print(f"[debug] init_network failed for ckpt={ckpt}: {e}")
                print(traceback.format_exc())
    # final attempt with default
    try:
        if debug:
            print(f"[debug] init_network(view='{view_name}') with default checkpoint")
        return NetworkView(view_name).init_network()
    except Exception as e:
        last_err = e
        if debug:
            print(f"[debug] init_network failed (default): {e}")
            print(traceback.format_exc())
        raise last_err


def load_connectome_w(ensemble_id: str, model_id: str, extent: int = 8, debug: bool = False):
    """
    Build a base Network with the requested extent, load the trained state into it,
    then compute p['w'] on that base net:
        w = syn_strength * syn_count * sign
    Returns:
        w : (n_edges, 1) float32
        edge_index : (2, n_edges) int64
    """
    import numpy as np
    import torch
    from flyvis import Network, NetworkView
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    # 1) init the trained model (native device)
    def _try(view):
        for ckpt in (0, -1, "best"):
            try:
                return NetworkView(view).init_network(checkpoint=ckpt)
            except Exception:
                pass
        return NetworkView(view).init_network()

    cand = []
    try:
        cand = [f"flow/{int(ensemble_id):04d}/{int(model_id):03d}",
                f"flow/{ensemble_id}/{int(model_id):03d}",
                f"flow/{int(ensemble_id):04d}/{model_id}",
                f"flow/{ensemble_id}/{model_id}"]
    except Exception:
        cand = [f"flow/{ensemble_id}/{model_id}"]

    trained = None
    for view in dict.fromkeys(cand):  # de-dup while keeping order
        try:
            trained = _try(view); break
        except Exception as e:
            last_err = e
    if trained is None:
        raise RuntimeError(f"Could not init NetworkView for {ensemble_id}/{model_id}: {last_err}")

    dev = next(trained.parameters()).device
    if debug: print(f"[debug] trained device = {dev}")

    # 2) build base net with desired extent (this sets n_edges ≈ 434,112 for extent=8)
    cfg = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    cfg.connectome.extent = extent
    net = Network(**cfg).to(dev)
    torch.set_grad_enabled(False)

    # 3) load trained state into base net (paramization is type/filter-based, so shapes match)
    net.load_state_dict(trained.state_dict())

    # 4) compute w on base net (COLUMN vector)
    params = net._param_api()
    w_t = params.edges.syn_strength * params.edges.syn_count * params.edges.sign
    w = (w_t.detach().to("cpu").to(torch.float32).reshape(-1, 1).numpy())  # (n_edges,1)

    # 5) edge index from base net (matches this extent)
    src = np.asarray(net.connectome.edges.source_index[:], dtype=np.int64)
    dst = np.asarray(net.connectome.edges.target_index[:], dtype=np.int64)
    edge_index = np.vstack([src, dst])

    # sanity
    if w.shape[0] != edge_index.shape[1]:
        raise ValueError(f"w rows {w.shape[0]} != n_edges {edge_index.shape[1]}")

    if debug:
        print(f"[debug] extent={extent}, n_edges={w.shape[0]} (w shape {w.shape}), edge_index {edge_index.shape}")

    return w, edge_index


def plot_umap2(save_path, emb2, labels):
    import matplotlib.pyplot as plt
    import numpy as np

    # --- sanitize emb2 to shape (n, 2) float32 ---
    arr = np.asarray(emb2)
    # If it's an object/ragged array (list of vectors), stack manually
    if arr.dtype == object or arr.ndim == 1:
        try:
            arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in arr])
        except Exception:
            raise ValueError(f"Embedding has invalid shape/dtype: {type(emb2)} {getattr(emb2, 'dtype', None)}")
    # Squeeze extraneous dims (e.g., (n,1,2))
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != 2:
        raise ValueError(f"Expected embedding of shape (n,2), got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)

    # --- ensure labels length matches n points (clip or pad) ---
    n = arr.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    else:
        # make it indexable and length n
        labels = list(labels)
        if len(labels) < n:
            labels = labels + ["" for _ in range(n - len(labels))]
        elif len(labels) > n:
            labels = labels[:n]

    # --- plot ---
    plt.figure(figsize=(6, 5))
    plt.scatter(arr[:, 0], arr[:, 1], s=36)
    # small text offset to avoid overlap
    dy = 0.015 * (float(np.ptp(arr[:, 1])) if n else 1.0)
    for i, lab in enumerate(labels):
        if lab:
            plt.text(arr[i, 0], arr[i, 1] + dy, str(lab), fontsize=7,
                     ha="center", va="bottom")
    plt.title("UMAP of FlyVis connectomes (p['w'])", fontsize=7)
    plt.xlabel("UMAP-1", fontsize=7); plt.ylabel("UMAP-2", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-id", type=str, default="0000")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end",   type=int, default=49)
    ap.add_argument("--extent", type=int, default=8)  # accepted but not compared
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", type=str, default="flyvis_connectomes")
    ap.add_argument("--skip-errors", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    mids = [f"{i:03d}" for i in range(args.start, args.end + 1)]
    W_rows, ok_ids = [], []
    first_ei = None

    print(f"Collecting p['w'] for ensemble={args.ensemble_id}, models {mids[0]}..{mids[-1]}")
    for m in mids:
        try:
            w, ei = load_connectome_w(args.ensemble_id, m, extent=args.extent, debug=args.debug)
            if first_ei is None:
                first_ei = ei
            elif ei.shape != first_ei.shape or not np.array_equal(ei, first_ei):
                print(f"[warn] edge_index differs for model {m}; proceeding anyway.")
            W_rows.append(w.reshape(1, -1)); ok_ids.append(m)
            print(f"  {m}: edges={w.size:,}")
        except Exception as e:
            print(f"[fail] {m}: {e}")
            if args.debug:
                print(traceback.format_exc())
            if args.skip_errors:
                continue
            else:
                sys.exit(1)

    if not W_rows:
        print("No models loaded; abort.")
        sys.exit(2)

    W = np.vstack(W_rows).astype(np.float32)
    print(f"Stacked W: {W.shape}  (~{W.nbytes/1e6:.1f} MB)")

    reducer = UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric=args.metric,
        random_state=args.seed,
        init="spectral",
        transform_mode="graph",
        verbose=True
    )
    reducer.fit_transform(W)

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{out}_edge_index.npz", edge_index=first_ei)
    np.savez_compressed(f"{out}_W.npz", W=W.astype(np.float32), model_ids=np.array(ok_ids, dtype="<U3"))

    dump(reducer, f"{out}_umap_model.joblib")

    # plot_umap2(Path(f"{out}_umap2d.png"), emb2, ok_ids)

    print("Saved:")
    print(" ", f"{out}_emb2.npz")
    print(" ", f"{out}_edge_index.npz")
    print(" ", f"{out}_umap2d.png")

if __name__ == "__main__":
    main()
