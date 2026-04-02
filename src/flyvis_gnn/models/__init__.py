# Lazy imports - individual modules can be imported directly
# e.g., from flyvis_gnn.models.graph_trainer import data_train
# Heavy imports deferred to avoid pulling in the full dependency chain on package load

__all__ = ["graph_trainer", "Siren_Network", "flyvis_gnn", "registry",
           "exploration_tree", "plot_exploration_tree", "plot_utils", "utils"]
