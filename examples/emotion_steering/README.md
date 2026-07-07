# Emotion Steering

Learn per-emotion steering vectors from a model's residual stream, evaluate them
on held-out stories, and chat with the model while steering it toward an emotion
on every decoding step.

Works with any mlx-lm or mlx-vlm chat model — Gemma 4 (e2b/e4b/12b/26b-a4b/31b),
Qwen 3.5/3.6, Llama, Mistral, etc. Models are loaded through `InterpretableModel`,
which tries mlx-lm first and falls back to mlx-vlm for multimodal repos.

## Learn vectors

```bash
uv run python examples/emotion_steering/run_experiment.py
# or pick a model
uv run python examples/emotion_steering/run_experiment.py -m mlx-community/Qwen3.5-27B-Instruct-4bit
```

The experiment:

1. Generates 50 matched training stories (5 emotions x 10 situations) plus held-out
   evaluation stories.
2. Captures a residual-stream proxy at ~2/3 model depth with `model.trace()`.
3. Averages activations from token 50 onward, subtracts the across-emotion mean,
   and projects out neutral-text principal components.
4. Reports held-out classification accuracy, logit-lens tokens, and steering deltas.

Artifacts land in `examples/emotion_steering/artifacts/<model-slug>_matched_50_story/`
(the default Gemma-4-e2b model uses plain `matched_50_story/`, which ships with the
repo so the chat CLI works out of the box).

## Steered chat

```bash
uv run python examples/emotion_steering/steered_chat_cli.py
```

Commands inside the chat: `/emotion <name|off>`, `/strength <float>`, `/status`,
`/reset`, `/emotions`, `/help`, `/quit`.

Steering is applied with the library's `model.steering()` context manager, which
patches the chosen layer so the intervention runs on every forward pass of the
token-by-token decoding loop:

```python
with model.steering({f"layers.{layer_idx}": iv.add_vector(unit_vector * strength)}):
    for token in generate(...):
        ...
```

Vectors are model-specific: chat with the same model you learned them with
(the CLI warns if they differ).
