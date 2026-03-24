# FlyVis-GNN

Graph neural networks recover interpretable circuit models from neural activity.

Synapse-level connectomes describe the structure of circuits, but not the electrical and chemical dynamics. Conversely, large-scale recordings of neural activity capture these dynamics, but not the circuit structure. We asked whether combining binary connectivity and recorded neural activity can be used to infer mechanistic models of neural circuits. We trained a graph neural network model (GNN) to forecast the activity of Drosophila visual system simulations. Trained on activity trajectories in response to visual inputs, the model recovers effective connectivity weights, neuron types, and nonlinear activation functions, even when 200% random connections are added to the adjacency matrix. Moreover, it correctly predicts causal effects of connection removal, demonstrating the ability to infer mechanistic dependencies directly from activity data. Our simple, flexible, and interpretable method recovers both structure and dynamics from incomplete anatomical reconstructions and activity.

The repository also includes an agentic workflow for hyper-parameter optimization in this ill-posed inverse problem.

**Project page:** [https://saalfeldlab.github.io/flyvis-gnn/](https://saalfeldlab.github.io/flyvis-gnn/)

## Installation

```bash
conda env create -f envs/environment.linux.yaml
conda activate flyvis-gnn
pip install -e .
```

The pretrained flyvis model (model 000, ~105 KB) is bundled in `assets/flyvis_model/` and used automatically.

Download the [DAVIS 2017](https://davischallenge.org/davis2017/code.html) dataset (480p) and set the environment variable:

```bash
export DATAVIS_ROOT=/path/to/DAVIS
```

The directory structure of the downloaded data will be:
```
${DATAVIS_ROOT}/
└── JPEGImages
    ├── 1080p
    │   ├── bear
    │   ├── blackswan
    ...
    └── 480p
        ├── bear
        ├── bike-trial
    ...
```

Trained GNN models and loss files are stored with [Git LFS](https://git-lfs.com/). After cloning, pull the model files:

```bash
git lfs install
git lfs pull
```

Simulation data must be generated first (Notebook 00.py) before training or testing.

## Usage

```bash
# Single training run
python GNN_Main.py -o generate_train_test_plot flyvis_noise_05

```
