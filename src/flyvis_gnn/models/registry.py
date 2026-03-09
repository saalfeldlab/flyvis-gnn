"""Model registry for flyvis-gnn.

Maps config signal_model_name strings (e.g. 'flyvis_A') to model classes.
Replaces scattered if/elif dispatch chains with a single lookup.

Usage:
    @register_model("flyvis_A", "flyvis_B", "flyvis_C")
    class FlyVisGNN(nn.Module):
        ...

    model = create_model("flyvis_A", config=config, device=device)
"""

_REGISTRY: dict[str, type] = {}
_discovered = False


def _discover_models():
    """Import all model modules so their @register_model decorators execute."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    import flyvis_gnn.models.flyvis_gnn  # noqa: F401 — triggers @register_model
    import flyvis_gnn.models.flyvis_linear  # noqa: F401 — triggers @register_model


def register_model(*names: str):
    """Class decorator that registers a model under one or more config names."""
    def decorator(cls):
        for name in names:
            if name in _REGISTRY:
                raise ValueError(
                    f"Model name '{name}' already registered to {_REGISTRY[name].__name__}"
                )
            _REGISTRY[name] = cls
        return cls
    return decorator


def create_model(name: str, **kwargs):
    """Look up model class by config name and instantiate it."""
    _discover_models()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    _discover_models()
    return sorted(_REGISTRY.keys())
