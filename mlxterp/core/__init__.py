"""Core components for mlxterp."""

from .activation import get_primary_tensor, is_array_like
from .proxy import ModuleProxy, OutputProxy, LayerListProxy, TraceContext
from .trace import Trace
from .intervention import (
    zero_out,
    scale,
    add_vector,
    replace_with,
    clamp,
    noise,
    InterventionComposer,
)
from .cache import ActivationCache, collect_activations
from .module_resolver import (
    ModuleResolver,
    normalize_layer_key,
    find_layer_key_pattern,
)

__all__ = [
    # Activation helpers
    "get_primary_tensor",
    "is_array_like",
    # Proxy
    "ModuleProxy",
    "OutputProxy",
    "LayerListProxy",
    "TraceContext",
    # Trace
    "Trace",
    # Intervention
    "zero_out",
    "scale",
    "add_vector",
    "replace_with",
    "clamp",
    "noise",
    "InterventionComposer",
    # Cache
    "ActivationCache",
    "collect_activations",
    # Module Resolution
    "ModuleResolver",
    "normalize_layer_key",
    "find_layer_key_pattern",
]
