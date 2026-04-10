"""
Regression tests for wrapped-model compatibility fixes.
"""

import mlx.core as mx
import mlx.nn as nn

from mlxterp import InterpretableModel, TunedLens
from mlxterp import interventions as iv
from mlxterp.core import normalize_layer_key


def _max_abs_diff(left: mx.array, right: mx.array) -> float:
    """Return the maximum absolute difference between two arrays."""
    diff = mx.max(mx.abs(left - right))
    mx.eval(diff)
    return float(diff)


class TupleLayer(nn.Module):
    """Simple layer that returns a hidden state plus auxiliary values."""

    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias

    def __call__(self, x: mx.array) -> tuple[mx.array, dict, int]:
        hidden = x + self.bias
        return hidden, {"bias": self.bias}, int(self.bias * 10)


class WrappedBackbone(nn.Module):
    """Backbone that mimics nested wrapper layouts used by converted models."""

    def __init__(self):
        super().__init__()
        self.layers = [TupleLayer(1.0), TupleLayer(2.0)]
        self.embed_tokens = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.per_layer_model_projection = nn.Linear(4, 4)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = x
        for layer in self.layers:
            hidden, _, _ = layer(hidden)
        return hidden


class LanguageModelWrapper(nn.Module):
    """Wrapper that exposes the common ``language_model.model`` nesting."""

    def __init__(self):
        super().__init__()
        self.model = WrappedBackbone()

    def __call__(self, x: mx.array) -> mx.array:
        return self.model(x)


class NestedWrappedModel(nn.Module):
    """Outer container with a ``language_model`` attribute."""

    def __init__(self):
        super().__init__()
        self.language_model = LanguageModelWrapper()

    def __call__(self, x: mx.array) -> mx.array:
        return self.language_model(x)


class GPTWrappedBackbone(nn.Module):
    """Backbone with GPT-style ``h`` layers under nested wrappers."""

    def __init__(self):
        super().__init__()
        self.h = [TupleLayer(1.0), TupleLayer(2.0)]
        self.wte = nn.Linear(4, 4)
        self.ln_f = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = x
        for layer in self.h:
            hidden, _, _ = layer(hidden)
        return hidden


class GPTLanguageModelWrapper(nn.Module):
    """Wrapper exposing nested GPT-style layers via ``language_model.model``."""

    def __init__(self):
        super().__init__()
        self.model = GPTWrappedBackbone()

    def __call__(self, x: mx.array) -> mx.array:
        return self.model(x)


class NestedWrappedGPTModel(nn.Module):
    """Outer container for nested GPT-style regression coverage."""

    def __init__(self):
        super().__init__()
        self.language_model = GPTLanguageModelWrapper()

    def __call__(self, x: mx.array) -> mx.array:
        return self.language_model(x)


class SimpleTokenizer:
    """Small tokenizer for regression tests that need text inputs."""

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        tokens = [ord(char) % self.vocab_size for char in text]
        return tokens or [0]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(ord("a") + (int(token) % 26)) for token in tokens)

    def tokenize(self, text: str) -> list[str]:
        return list(text) if text else [""]


class AnalysisBackbone(nn.Module):
    """Backbone with tuple-returning layers and a real output projection."""

    def __init__(self, vocab_size: int = 16, hidden_dim: int = 8):
        super().__init__()
        self.layers = [TupleLayer(0.5), TupleLayer(1.0)]
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.per_layer_model_projection = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden, _, _ = layer(hidden)
        hidden = self.norm(hidden)
        return self.per_layer_model_projection(hidden)


class AnalysisLanguageModelWrapper(nn.Module):
    """Wrapper that exposes ``language_model.model`` for analysis tests."""

    def __init__(self):
        super().__init__()
        self.model = AnalysisBackbone()

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.model(tokens)


class AnalysisWrappedModel(nn.Module):
    """Outer wrapped model used by text-based regression tests."""

    def __init__(self):
        super().__init__()
        self.language_model = AnalysisLanguageModelWrapper()

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.language_model(tokens)


class GemmaLikeBackbone(nn.Module):
    """Backbone with a non-vocab internal projection plus tied embeddings."""

    def __init__(self, vocab_size: int = 16, hidden_dim: int = 8, projection_dim: int = 12):
        super().__init__()
        self.layers = [TupleLayer(0.5), TupleLayer(1.0)]
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.per_layer_model_projection = nn.Linear(hidden_dim, projection_dim)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden, _, _ = layer(hidden)
        hidden = self.norm(hidden)
        return hidden @ self.embed_tokens.weight.T


class GemmaLikeLanguageModelWrapper(nn.Module):
    """Wrapper with tied embeddings and a nested text backbone."""

    def __init__(self):
        super().__init__()
        self.model = GemmaLikeBackbone()
        self.tie_word_embeddings = True

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.model(tokens)


class GemmaLikeWrappedModel(nn.Module):
    """Outer wrapped model that mimics Gemma-style nesting."""

    def __init__(self):
        super().__init__()
        self.language_model = GemmaLikeLanguageModelWrapper()

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.language_model(tokens)


def test_replace_with_tuple_preserves_auxiliary_outputs():
    """`replace_with` should patch tuple outputs without clobbering aux values."""
    target = (mx.zeros((1, 5, 4)), {"cache": "original"}, 3)
    replacement = (mx.full((1, 3, 4), 7.0), {"cache": "replacement"})
    expected = mx.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [7.0, 7.0, 7.0, 7.0],
                [7.0, 7.0, 7.0, 7.0],
                [7.0, 7.0, 7.0, 7.0],
            ]
        ]
    )

    patched = iv.replace_with(replacement, align="end")(target)

    assert isinstance(patched, tuple)
    assert patched[1] == target[1]
    assert patched[2] == target[2]
    assert _max_abs_diff(patched[0], expected) == 0.0


def test_nested_wrapped_models_resolve_layers_and_projection_paths():
    """Nested wrapper layouts should resolve layers and canonicalize names."""
    model = InterpretableModel(NestedWrappedModel())

    assert len(model.layers) == 2
    assert model._module_resolver.get_embedding_path() == "language_model.model.embed_tokens"
    assert model._module_resolver.get_final_norm_path() == "language_model.model.norm"
    assert model._module_resolver.get_output_projection()[1:] == (
        "language_model.model.per_layer_model_projection",
        False,
    )
    assert (
        normalize_layer_key("model.language_model.model.layers.0.self_attn") == "layers.0.self_attn"
    )


def test_wrapped_tuple_layers_support_proxy_access_and_interventions():
    """Proxy lookups and interventions should work through wrapped tuple layers."""
    model = InterpretableModel(NestedWrappedModel())
    inputs = mx.zeros((1, 2, 4))

    with model.trace(inputs) as trace:
        layer_0 = model.layers[0].output.save()
        baseline_output = model.output.save()

    with model.trace(inputs, interventions={"layers.0": iv.scale(0.0)}):
        modified_layer_0 = model.layers[0].output.save()
        modified_output = model.output.save()

    assert "model.language_model.model.layers.0" in trace.activations
    assert isinstance(layer_0, tuple)
    assert isinstance(modified_layer_0, tuple)
    assert _max_abs_diff(layer_0[0], mx.ones((1, 2, 4))) == 0.0
    assert _max_abs_diff(modified_layer_0[0], mx.zeros((1, 2, 4))) == 0.0
    assert _max_abs_diff(baseline_output, mx.full((1, 2, 4), 3.0)) == 0.0
    assert _max_abs_diff(modified_output, mx.full((1, 2, 4), 2.0)) == 0.0


def test_nested_wrapped_gpt_layers_resolve_custom_layer_attr():
    """Nested wrapper layouts should also resolve non-default layer attributes."""
    model = InterpretableModel(NestedWrappedGPTModel(), layer_attr="h")
    inputs = mx.zeros((1, 2, 4))

    assert len(model.layers) == 2

    with model.trace(inputs) as trace:
        layer_0 = model.layers[0].output.save()
        baseline_output = model.output.save()

    with model.trace(inputs, interventions={"h.0": iv.scale(0.0)}):
        modified_layer_0 = model.layers[0].output.save()
        modified_output = model.output.save()

    assert "model.language_model.model.h.0" in trace.activations
    assert isinstance(layer_0, tuple)
    assert isinstance(modified_layer_0, tuple)
    assert _max_abs_diff(layer_0[0], mx.ones((1, 2, 4))) == 0.0
    assert _max_abs_diff(modified_layer_0[0], mx.zeros((1, 2, 4))) == 0.0
    assert _max_abs_diff(baseline_output, mx.full((1, 2, 4), 3.0)) == 0.0
    assert _max_abs_diff(modified_output, mx.full((1, 2, 4), 2.0)) == 0.0


def test_logit_lens_handles_wrapped_tuple_layer_outputs():
    """Logit lens should unwrap tuple activations from wrapped models."""
    model = InterpretableModel(AnalysisWrappedModel(), tokenizer=SimpleTokenizer())

    results = model.logit_lens("hello", layers=[0], top_k=1)

    assert 0 in results
    assert len(results[0]) == len(model.encode("hello"))
    assert len(results[0][0]) == 1


def test_train_tuned_lens_handles_wrapped_tuple_layer_outputs():
    """Tuned lens training should infer hidden size from tuple activations."""
    model = InterpretableModel(AnalysisWrappedModel(), tokenizer=SimpleTokenizer())

    tuned_lens = model.train_tuned_lens(
        ["hello world " * 5],
        num_steps=1,
        max_seq_len=10,
        verbose=False,
    )

    assert tuned_lens.hidden_dim == 8
    assert tuned_lens.num_layers == 2


def test_tuned_lens_handles_wrapped_tuple_layer_outputs():
    """Tuned lens analysis should unwrap tuple activations from wrapped models."""
    model = InterpretableModel(AnalysisWrappedModel(), tokenizer=SimpleTokenizer())
    tuned_lens = TunedLens(num_layers=2, hidden_dim=8)

    results = model.tuned_lens("hello", tuned_lens, layers=[0], top_k=1)

    assert 0 in results
    assert len(results[0]) == len(model.encode("hello"))
    assert len(results[0][0]) == 1


def test_tied_wrapped_models_ignore_internal_projection_for_output_head():
    """Tied models should prefer embeddings over non-vocab internal projections."""
    model = InterpretableModel(GemmaLikeWrappedModel(), tokenizer=SimpleTokenizer())

    projection, projection_path, is_weight_tied = model._module_resolver.get_output_projection()

    assert projection is model._module_resolver.get_embedding_layer()
    assert projection_path == "language_model.model.embed_tokens"
    assert is_weight_tied is True

    tuned_lens = model.train_tuned_lens(
        ["hello world " * 5],
        num_steps=1,
        max_seq_len=10,
        verbose=False,
    )

    assert tuned_lens.hidden_dim == 8
