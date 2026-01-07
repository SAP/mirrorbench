# Ensure environment variables from .env are loaded once the package is imported
from mirrorbench.core.env import ensure_env_loaded as _ensure_env_loaded

_ensure_env_loaded()

# Ensure built-in model clients and adapters register on import
from mirrorbench import (  # noqa: F401, E402
    adapters as _adapters,
    datasets as _datasets,
    metrics as _metrics,
    model_clients as _model_clients,
    tasks as _tasks,
)
from mirrorbench.core.registry import registry  # noqa: E402

__all__ = ["registry"]
