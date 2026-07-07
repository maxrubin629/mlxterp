"""
Tests for dict/kwargs trace inputs and model output normalization.

These cover the mlx-vlm style calling convention where models take keyword
inputs (input_ids, pixel_values, mask) and return an object with `.logits`.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxterp import InterpretableModel
from mlxterp import interventions as iv


class LogitsOutput:
    """Mimics mlx-vlm's LanguageModelOutput container."""

    def __init__(self, logits: mx.array):
        self.logits = logits


class OffsetBlock(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.offset


class VLMStyleModel(nn.Module):
    """Accepts keyword inputs and returns an object with `.logits`."""

    def __init__(self, vocab_size: int = 10, hidden_dim: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [OffsetBlock(1.0), OffsetBlock(2.0)]

    def __call__(self, input_ids=None, pixel_values=None, mask=None, cache=None):
        hidden = self.embed_tokens(input_ids)
        if mask is not None:
            hidden = hidden + mask[..., None].astype(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden)
        return LogitsOutput(logits=hidden)


class TupleOutputModel(nn.Module):
    """Returns a (logits, aux) tuple from the top-level forward."""

    def __init__(self, vocab_size: int = 10, hidden_dim: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [OffsetBlock(1.0)]

    def __call__(self, tokens: mx.array):
        hidden = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden, {"cache": "aux"}


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)))


def test_dict_inputs_call_model_with_kwargs():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    with model.trace({"input_ids": input_ids}) as trace:
        pass

    assert isinstance(trace.output, mx.array)
    assert trace.output.shape == (1, 3, 4)
    assert isinstance(trace.raw_output, LogitsOutput)


def test_keyword_inputs_call_model_with_kwargs():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    with model.trace(input_ids=input_ids) as trace:
        saved = model.output.save()

    assert isinstance(trace.output, mx.array)
    assert _max_abs_diff(saved, trace.output) < 1e-5


def test_attention_mask_is_adapted_to_mask_argument():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])
    mask = mx.array([[0, 1, 1]])

    with model.trace({"input_ids": input_ids, "mask": mask}) as direct:
        pass
    with model.trace({"input_ids": input_ids, "attention_mask": mask}) as adapted:
        pass

    assert _max_abs_diff(direct.output, adapted.output) < 1e-5

    with model.trace({"input_ids": input_ids}) as unmasked:
        pass

    assert _max_abs_diff(adapted.output, unmasked.output) > 1e-3


def test_passing_both_inputs_and_keyword_inputs_raises():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="not both"):
        with model.trace({"input_ids": input_ids}, input_ids=input_ids):
            pass


def test_trace_without_any_inputs_raises():
    model = InterpretableModel(VLMStyleModel())

    with pytest.raises(ValueError, match="requires inputs"):
        with model.trace():
            pass


def test_tuple_model_output_is_normalized_to_primary_tensor():
    model = InterpretableModel(TupleOutputModel())
    tokens = mx.array([[1, 2, 3]])

    with model.trace(tokens) as trace:
        pass

    assert isinstance(trace.output, mx.array)
    assert isinstance(trace.raw_output, tuple)
    assert trace.raw_output[1] == {"cache": "aux"}


def test_interventions_apply_with_keyword_inputs():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    with model.trace(input_ids=input_ids) as baseline:
        pass
    with model.trace(
        {"input_ids": input_ids},
        interventions={"layers.0": iv.add_vector(mx.full((4,), 5.0))},
    ) as steered:
        pass

    assert _max_abs_diff(steered.output, baseline.output + 5.0) < 1e-5


def test_trace_get_and_get_activation_accept_shorthand_names():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    with model.trace(input_ids=input_ids) as trace:
        model.layers[0].output.save()

    assert trace.get("layers.0.output") is not None
    assert trace.get_activation("layers.0") is not None
    assert trace.get_activation("model.layers.0") is not None
    assert trace.get_activation("layers.7") is None


def test_steering_context_applies_interventions_across_forward_passes():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    baseline = model(input_ids=input_ids).logits

    with model.steering({"layers.0": iv.add_vector(mx.full((4,), 5.0))}):
        first = model(input_ids=input_ids).logits
        second = model(input_ids=input_ids).logits

    restored = model(input_ids=input_ids).logits

    assert _max_abs_diff(first, baseline + 5.0) < 1e-5
    assert _max_abs_diff(second, baseline + 5.0) < 1e-5
    assert _max_abs_diff(restored, baseline) < 1e-5


def test_steering_context_with_no_interventions_is_a_no_op():
    model = InterpretableModel(VLMStyleModel())
    input_ids = mx.array([[1, 2, 3]])

    baseline = model(input_ids=input_ids).logits

    with model.steering():
        unchanged = model(input_ids=input_ids).logits

    assert _max_abs_diff(unchanged, baseline) < 1e-5
