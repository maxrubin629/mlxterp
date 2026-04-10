"""
Helpers for working with activation values.

Some wrapped MLX language models return tuples such as
``(hidden_states, kv_cache, offset)`` from intermediate blocks. Most mlxterp
analysis utilities operate on the hidden-state tensor, so these helpers unwrap
that primary tensor when needed.
"""

from typing import Any


def is_array_like(value: Any) -> bool:
    """Return True for MLX tensor-like values we can operate on directly."""
    return hasattr(value, "shape") and hasattr(value, "tolist")


def get_primary_tensor(value: Any) -> Any:
    """
    Return the primary tensor from an activation or model output.

    Args:
        value: Tensor-like value, or a tuple/list containing one.

    Returns:
        The tensor-like value to treat as the activation.

    Raises:
        TypeError: If no tensor-like value is present.
    """
    if is_array_like(value):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            if is_array_like(item):
                return item

    raise TypeError(
        f"Activation value must be tensor-like or contain a tensor-like value: {type(value)!r}"
    )
