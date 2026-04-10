"""
Intervention utilities for modifying activations during forward passes.

Provides helper functions for common intervention patterns.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, cast

import mlx.core as mx

from .activation import get_primary_tensor, is_array_like


def _apply_to_primary_tensor(value: Any, transform: Callable[[mx.array], mx.array]) -> Any:
    """
    Apply an intervention to the primary hidden-state tensor.

    Many language-model blocks return a bare tensor, but some architectures
    return tuples like ``(hidden_states, kv_cache, offset)``. For those cases
    we intervene on the first tensor-like element and preserve the rest.
    """
    if is_array_like(value):
        return transform(value)

    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if is_array_like(item):
                items[index] = transform(item)
                return tuple(items)
        raise TypeError(f"Tuple output does not contain a tensor-like value: {type(value)!r}")

    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if is_array_like(item):
                items[index] = transform(item)
                return items
        raise TypeError(f"List output does not contain a tensor-like value: {type(value)!r}")

    raise TypeError(f"Unsupported activation type for intervention: {type(value)!r}")


def zero_out(x: Any) -> Any:
    """
    Zero out an activation.

    Example:
        with model.trace(input, interventions={"layers.4": zero_out}):
            ...
    """
    return _apply_to_primary_tensor(x, mx.zeros_like)


def scale(factor: float) -> Callable[[Any], Any]:
    """
    Scale an activation by a constant factor.

    Args:
        factor: Scaling factor

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": scale(0.5)}):
            ...
    """

    def _scale(x: Any) -> Any:
        return _apply_to_primary_tensor(x, lambda tensor: tensor * factor)

    return _scale


def add_vector(vector: mx.array) -> Callable[[Any], Any]:
    """
    Add a vector to an activation (steering vector).

    Args:
        vector: Vector to add (must be broadcastable)

    Returns:
        Intervention function

    Example:
        steering_vec = mx.random.normal((hidden_dim,))
        with model.trace(input, interventions={"layers.4": add_vector(steering_vec)}):
            ...
    """

    def _add(x: Any) -> Any:
        return _apply_to_primary_tensor(x, lambda tensor: tensor + vector)

    return _add


def replace_with(value: Union[mx.array, float], align: str = "end") -> Callable[[Any], Any]:
    """
    Replace activation with a fixed value.

    Args:
        value: Replacement value (array or scalar)
        align: How to align when sequence lengths differ:
            - "end": Align at end (last tokens match) - default, best for activation patching
            - "start": Align at start (first tokens match)
            - "strict": Raise error if shapes don't match

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": replace_with(0.0)}):
            ...

        # Activation patching with different sequence lengths:
        with model.trace(clean_text) as trace:
            clean_act = trace.activations["model.model.layers.10.mlp"]
        with model.trace(corrupted_text,
                        interventions={"layers.10.mlp": replace_with(clean_act)}):
            patched = model.output.save()
    """

    def _replace_tensor(x: mx.array) -> mx.array:
        if isinstance(value, (int, float)):
            return mx.full(x.shape, value)

        replacement = get_primary_tensor(value)

        # If shapes match exactly, use the value directly
        if replacement.shape == x.shape:
            return cast(mx.array, replacement)

        # Handle sequence length mismatch (common in activation patching)
        # Assuming shape is (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
        if replacement.ndim >= 2 and x.ndim >= 2:
            # Get sequence dimension (usually axis 1 for 3D, axis 0 for 2D)
            seq_axis = 1 if replacement.ndim == 3 else 0
            value_seq_len = replacement.shape[seq_axis]
            x_seq_len = x.shape[seq_axis]

            if value_seq_len != x_seq_len:
                if align == "strict":
                    raise ValueError(
                        "Shape mismatch: replacement has shape "
                        f"{replacement.shape} but target has shape {x.shape}. "
                        "Use align='end' or align='start' to handle different "
                        "sequence lengths."
                    )

                # Create output with same shape as x
                result = x.copy() if hasattr(x, "copy") else mx.array(x)

                if align == "end":
                    # Align at end: patch last min(value_seq_len, x_seq_len) tokens
                    min_len = min(value_seq_len, x_seq_len)
                    if replacement.ndim == 3:
                        result[:, -min_len:, :] = replacement[:, -min_len:, :]
                    else:
                        result[-min_len:, :] = replacement[-min_len:, :]
                else:  # align == "start"
                    # Align at start: patch first min(value_seq_len, x_seq_len) tokens
                    min_len = min(value_seq_len, x_seq_len)
                    if replacement.ndim == 3:
                        result[:, :min_len, :] = replacement[:, :min_len, :]
                    else:
                        result[:min_len, :] = replacement[:min_len, :]

                return result

        # Fallback: try to broadcast (will fail if truly incompatible)
        return mx.broadcast_to(replacement, x.shape)

    def _replace(x: Any) -> Any:
        return _apply_to_primary_tensor(x, _replace_tensor)

    return _replace


def clamp(min_val: Optional[float] = None, max_val: Optional[float] = None) -> Callable[[Any], Any]:
    """
    Clamp activation values to a range.

    Args:
        min_val: Minimum value (None for no minimum)
        max_val: Maximum value (None for no maximum)

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": clamp(min_val=-1, max_val=1)}):
            ...
    """

    def _clamp(x: Any) -> Any:
        def _clamp_tensor(tensor: mx.array) -> mx.array:
            result = tensor
            if min_val is not None:
                result = mx.maximum(result, min_val)
            if max_val is not None:
                result = mx.minimum(result, max_val)
            return result

        return _apply_to_primary_tensor(x, _clamp_tensor)

    return _clamp


def noise(std: float = 0.1) -> Callable[[Any], Any]:
    """
    Add Gaussian noise to an activation.

    Args:
        std: Standard deviation of noise

    Returns:
        Intervention function

    Example:
        with model.trace(input, interventions={"layers.4": noise(std=0.1)}):
            ...
    """

    def _noise(x: Any) -> Any:
        return _apply_to_primary_tensor(
            x,
            lambda tensor: tensor + mx.random.normal(tensor.shape) * std,
        )

    return _noise


class InterventionComposer:
    """
    Compose multiple interventions into a single function.

    Example:
        combined = InterventionComposer() \\
            .add(scale(0.5)) \\
            .add(add_vector(steering_vec)) \\
            .build()

        with model.trace(input, interventions={"layers.4": combined}):
            ...
    """

    def __init__(self):
        self.interventions = []

    def add(self, fn: Callable[[Any], Any]) -> "InterventionComposer":
        """Add an intervention to the composition"""
        self.interventions.append(fn)
        return self

    def build(self) -> Callable[[Any], Any]:
        """Build the composed intervention function"""

        def _composed(x: Any) -> Any:
            result = x
            for fn in self.interventions:
                result = fn(result)
            return result

        return _composed
