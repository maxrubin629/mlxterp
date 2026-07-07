# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- mlx-vlm loading fallback in `InterpretableModel`: multimodal repos such as Gemma 4 now load by name, with the processor exposed as `model.processor` and its tokenizer reused automatically.
- Dict and keyword trace inputs: `model.trace({"input_ids": ...})` and `model.trace(input_ids=..., attention_mask=...)` call the wrapped model with keyword arguments, translating `attention_mask` to `mask` when the model's signature expects it.
- Output normalization: `trace.output` now yields the logits tensor for models that return `.logits` containers or tuples; the untouched value is available as `trace.raw_output`.
- `InterpretableModel.steering()` context manager for applying interventions across every forward pass in a block (e.g., token-by-token generation loops).
- `trace.get()` / `trace.get_activation()` accept shorthand names like `layers.3` for wrapped-model keys.
- Emotion-steering example (`examples/emotion_steering/`): learns per-emotion residual-stream vectors, evaluates them on held-out stories, and ships an interactive steered chat CLI. Works with mlx-lm and mlx-vlm models.
- Exported `get_primary_tensor` / `is_array_like` from `mlxterp.core`.

### Changed

- `requires-python` is now `>=3.10` (required by mlx-vlm).

### Fixed

- `get_activations` unwraps tuple layer outputs from wrapped models before position indexing.
- Test suite runs headless (`tests/conftest.py` forces the Agg matplotlib backend); plotting tests previously blocked on a native window.
- SAE integration tests referenced a nonexistent HF repo (`mlx-community/Llama-3.2-1B-Instruct`) and matched MLP subprojection keys instead of the MLP output.

- Improved wrapped-model compatibility for tracing and analysis utilities.
- Added support for tuple-style layer outputs by consistently operating on the primary tensor while preserving auxiliary values.
- Improved module and layer resolution for nested wrapper layouts such as `language_model.model.*`, including non-default layer attributes like `h`.
- Added regression coverage for wrapped Llama-, Qwen-, and Gemma-style compatibility paths.
