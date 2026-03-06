# FlyVis-GNN

Graph neural networks for fly visual system connectivity recovery, with agentic hyper-parameter optimization.

**Project page:** [https://saalfeldlab.github.io/flyvis-gnn/](https://saalfeldlab.github.io/flyvis-gnn/)

## Installation

```bash
conda env create -f envs/environment.linux.yaml
conda activate flyvis-gnn
```

## Usage

```bash
# Single training run
python GNN_Main.py -o generate_train_test_plot flyvis_62_0

# Agentic hyper-parameter optimization
python GNN_LLM_parallel_flyvis.py -o train_test_plot_Claude_cluster flyvis_62_0 iterations=144 --resume
```
