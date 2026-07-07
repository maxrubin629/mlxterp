#!/usr/bin/env python3
"""
Interactive steered chat CLI for learned emotion vectors.

This script loads the learned emotion vectors from the local example artifacts
and applies one selected vector on every decoding step during generation.
Works with any mlx-lm or mlx-vlm chat model (Gemma 4, Qwen 3.5/3.6, Llama, ...).

Usage:
    uv run python examples/emotion_steering/steered_chat_cli.py
    uv run python examples/emotion_steering/steered_chat_cli.py -m <model-repo> --artifact-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np

from mlxterp import InterpretableModel
from mlxterp import interventions as iv

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_ARTIFACT_BASENAME = "matched_50_story"
DEFAULT_LAYER = 23
DEFAULT_EMOTION = "off"
DEFAULT_STRENGTH = 8.0
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 64
DEFAULT_TOP_P = 0.95


@dataclass
class SteeringArtifacts:
    artifact_dir: Path
    summary: dict[str, Any]
    layer_idx: int
    emotions: list[str]
    unit_vectors_np: dict[str, np.ndarray]
    unit_vectors_mx: dict[str, mx.array]


@dataclass
class ChatState:
    active_emotion: str
    strength: float
    messages: list[dict[str, str]] = field(default_factory=list)


def model_slug(model_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in model_name).strip("_")


def default_artifact_dir_for_model(model_name: str) -> Path:
    if model_name == DEFAULT_MODEL:
        return SCRIPT_DIR / "artifacts" / DEFAULT_ARTIFACT_BASENAME

    return SCRIPT_DIR / "artifacts" / f"{model_slug(model_name)}_{DEFAULT_ARTIFACT_BASENAME}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Model repo to load for chat.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifact directory containing summary.json and emotion_vectors.npz. Defaults to a model-specific example path.",
    )
    parser.add_argument(
        "--emotion",
        default=DEFAULT_EMOTION,
        help="Initial steering emotion name, or 'off'.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_STRENGTH,
        help="Initial steering strength multiplier.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to decode per assistant turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k sampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p sampling.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive.")
    if not math.isfinite(args.strength):
        raise SystemExit("--strength must be a finite float.")
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0.")
    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0.")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in the range (0, 1].")


def load_artifacts(artifact_dir: Path) -> SteeringArtifacts:
    summary_path = artifact_dir / "summary.json"
    vectors_path = artifact_dir / "emotion_vectors.npz"

    if not artifact_dir.exists():
        raise SystemExit(
            f"Artifact directory does not exist: {artifact_dir}\n"
            "Run examples/emotion_steering/run_experiment.py first to learn vectors for this model."
        )
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json in artifact directory: {artifact_dir}")
    if not vectors_path.exists():
        raise SystemExit(f"Missing emotion_vectors.npz in artifact directory: {artifact_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    emotions = summary.get("emotions")
    if not isinstance(emotions, list) or not emotions:
        raise SystemExit(f"summary.json is missing a usable 'emotions' list: {summary_path}")

    layer_idx = int(summary.get("layer_idx", DEFAULT_LAYER))

    npz = np.load(vectors_path, allow_pickle=True)
    unit_vectors_np: dict[str, np.ndarray] = {}
    unit_vectors_mx: dict[str, mx.array] = {}
    hidden_dim = None

    for emotion in emotions:
        key = f"unit_vector_{emotion}"
        if key not in npz:
            raise SystemExit(f"emotion_vectors.npz is missing '{key}'.")

        vector = np.array(npz[key], dtype=np.float32)
        if vector.ndim != 1:
            raise SystemExit(f"Expected '{key}' to be 1D, got shape {vector.shape}.")
        if hidden_dim is None:
            hidden_dim = vector.shape[0]
        elif vector.shape[0] != hidden_dim:
            raise SystemExit(
                f"Inconsistent vector sizes in emotion_vectors.npz: "
                f"expected {hidden_dim}, got {vector.shape[0]} for '{key}'."
            )

        unit_vectors_np[emotion] = vector
        unit_vectors_mx[emotion] = mx.array(vector)

    return SteeringArtifacts(
        artifact_dir=artifact_dir,
        summary=summary,
        layer_idx=layer_idx,
        emotions=emotions,
        unit_vectors_np=unit_vectors_np,
        unit_vectors_mx=unit_vectors_mx,
    )


def normalize_emotion_name(name: str) -> str:
    return name.strip().lower()


def validate_emotion_or_exit(emotion: str, artifacts: SteeringArtifacts) -> str:
    normalized = normalize_emotion_name(emotion)
    if normalized == "off":
        return normalized
    if normalized not in artifacts.emotions:
        available = ", ".join(artifacts.emotions)
        raise SystemExit(f"Unknown emotion '{emotion}'. Available emotions: {available}, off")
    return normalized


def print_banner(
    model_name: str,
    artifacts: SteeringArtifacts,
    state: ChatState,
    args: argparse.Namespace,
) -> None:
    available = ", ".join(artifacts.emotions)
    print("Steered Chat CLI")
    print(f"Model: {model_name}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Layer: {artifacts.layer_idx}")
    print(
        "Sampling: "
        f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(f"Available emotions: {available}")
    print(f"Active steering: {state.active_emotion}")
    print(f"Strength: {state.strength}")
    print("Type /help for commands.")
    print()


def print_help(artifacts: SteeringArtifacts) -> None:
    available = ", ".join(artifacts.emotions)
    print("Commands:")
    print("  /emotion <name|off>  Set active steering emotion.")
    print("  /strength <float>    Set steering strength for future turns.")
    print("  /status              Show current chat and steering settings.")
    print("  /reset               Clear conversation history.")
    print("  /emotions            List available emotions.")
    print("  /help                Show this help.")
    print("  /quit                Exit the CLI.")
    print(f"Available emotions: {available}")


def print_status(
    model_name: str,
    artifacts: SteeringArtifacts,
    state: ChatState,
    args: argparse.Namespace,
) -> None:
    user_turns = sum(1 for message in state.messages if message["role"] == "user")
    assistant_turns = sum(1 for message in state.messages if message["role"] == "assistant")
    print("Status")
    print(f"  Model: {model_name}")
    print(f"  Artifact dir: {artifacts.artifact_dir}")
    print(f"  Layer: {artifacts.layer_idx}")
    print(f"  Active emotion: {state.active_emotion}")
    print(f"  Strength: {state.strength}")
    print(
        "  Sampling: "
        f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(f"  Turns: user={user_turns}, assistant={assistant_turns}")


def has_chat_template(template_owner: object | None) -> bool:
    return bool(
        template_owner is not None
        and hasattr(template_owner, "apply_chat_template")
        and getattr(template_owner, "chat_template", None)
    )


def build_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    processor: Any = None,
) -> str:
    template_owner = processor if has_chat_template(processor) else tokenizer
    if has_chat_template(template_owner):
        return template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def collect_stop_token_ids(model: InterpretableModel) -> set[int]:
    """
    Collect end-of-sequence/turn token ids.

    Deliberately restricted to declared EOS tokens: chat formats like Gemma 4
    emit special tokens mid-generation (e.g., thinking-channel markers), so
    treating every special token as a stop would truncate replies.
    """
    stop_ids: set[int] = set()

    tokenizer_eos = getattr(model.tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        stop_ids.add(int(tokenizer_eos))

    # mlx-lm's TokenizerWrapper exposes the full stop set (eos + end-of-turn)
    tokenizer_eos_ids = getattr(model.tokenizer, "eos_token_ids", None)
    if tokenizer_eos_ids is not None:
        stop_ids.update(int(token_id) for token_id in tokenizer_eos_ids)

    # mlx-vlm models expose `config`, mlx-lm models expose `args`
    model_config = getattr(model.model, "config", None)
    if model_config is None:
        model_config = getattr(model.model, "args", None)
    model_eos = getattr(model_config, "eos_token_id", None) if model_config is not None else None
    if isinstance(model_eos, (list, tuple)):
        stop_ids.update(int(token_id) for token_id in model_eos)
    elif model_eos is not None:
        stop_ids.add(int(model_eos))

    return stop_ids


def build_interventions(
    artifacts: SteeringArtifacts,
    active_emotion: str,
    strength: float,
) -> dict[str, Any] | None:
    if active_emotion == "off":
        return None

    steering_vector = artifacts.unit_vectors_mx[active_emotion] * strength
    return {f"layers.{artifacts.layer_idx}": iv.add_vector(steering_vector)}


def token_id_to_int(token: Any) -> int:
    if hasattr(token, "tolist"):
        value = token.tolist()
    else:
        value = token

    while isinstance(value, list):
        if not value:
            raise ValueError("Cannot convert empty token list to integer.")
        value = value[0]

    return int(value)


def stream_tokens_vlm(
    model: InterpretableModel,
    prompt_text: str,
    args: argparse.Namespace,
) -> Iterator[int]:
    """Yield generated token ids from an mlx-vlm loaded model."""
    from mlx_vlm import prepare_inputs
    from mlx_vlm.generate import generate_step

    prepared = prepare_inputs(
        model.processor,
        prompts=prompt_text,
        return_tensors="mlx",
    )
    input_ids = prepared["input_ids"]
    pixel_values = prepared.get("pixel_values")
    mask = prepared.get("attention_mask", prepared.get("mask"))

    extra_kwargs = {
        key: value
        for key, value in prepared.items()
        if key not in {"input_ids", "pixel_values", "attention_mask", "mask"}
    }

    for token, _logprobs in generate_step(
        input_ids,
        model.model,
        pixel_values=pixel_values,
        mask=mask,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        **extra_kwargs,
    ):
        yield token_id_to_int(token)


def stream_tokens_lm(
    model: InterpretableModel,
    prompt_text: str,
    args: argparse.Namespace,
) -> Iterator[int]:
    """Yield generated token ids from an mlx-lm loaded model."""
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)
    for response in stream_generate(
        model.model,
        model.tokenizer,
        prompt=prompt_text,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
    ):
        yield response.token


def stream_reply(
    model: InterpretableModel,
    artifacts: SteeringArtifacts,
    state: ChatState,
    args: argparse.Namespace,
    stop_token_ids: set[int],
) -> str:
    prompt_text = build_chat_prompt(model.tokenizer, state.messages, processor=model.processor)
    generated_tokens: list[int] = []
    printed_text = ""
    interventions = build_interventions(artifacts, state.active_emotion, state.strength)

    if getattr(model, "processor", None) is not None:
        token_stream = stream_tokens_vlm(model, prompt_text, args)
    else:
        token_stream = stream_tokens_lm(model, prompt_text, args)

    print("assistant> ", end="", flush=True)
    with model.steering(interventions):
        for next_token in token_stream:
            if next_token in stop_token_ids:
                break

            generated_tokens.append(next_token)
            decoded_text = model.decode(generated_tokens)
            if decoded_text.startswith(printed_text):
                delta = decoded_text[len(printed_text):]
            else:
                delta = decoded_text
            if delta:
                print(delta, end="", flush=True)
            printed_text = decoded_text

    print()
    return printed_text.strip()


def handle_command(
    command: str,
    model_name: str,
    artifacts: SteeringArtifacts,
    state: ChatState,
    args: argparse.Namespace,
) -> bool:
    raw = command.strip()
    if raw == "/quit":
        return False
    if raw == "/help":
        print_help(artifacts)
        return True
    if raw == "/status":
        print_status(model_name, artifacts, state, args)
        return True
    if raw == "/reset":
        state.messages.clear()
        print("Conversation history cleared.")
        return True
    if raw == "/emotions":
        print("Available emotions:", ", ".join(artifacts.emotions))
        print("Use '/emotion off' to disable steering.")
        return True
    if raw.startswith("/emotion"):
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: /emotion <name|off>")
            return True

        emotion = normalize_emotion_name(parts[1])
        if emotion != "off" and emotion not in artifacts.emotions:
            print(f"Unknown emotion '{parts[1]}'. Available: {', '.join(artifacts.emotions)}, off")
            return True

        state.active_emotion = emotion
        print(f"Active emotion set to {state.active_emotion}.")
        return True
    if raw.startswith("/strength"):
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: /strength <float>")
            return True
        try:
            strength = float(parts[1])
        except ValueError:
            print(f"Could not parse strength value: {parts[1]}")
            return True
        if not math.isfinite(strength):
            print("Strength must be a finite float.")
            return True

        state.strength = strength
        print(f"Steering strength set to {state.strength}.")
        return True

    print(f"Unknown command: {raw}")
    print("Type /help for available commands.")
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)

    artifact_dir = (
        Path(args.artifact_dir).expanduser().resolve()
        if args.artifact_dir is not None
        else default_artifact_dir_for_model(args.model)
    )
    artifacts = load_artifacts(artifact_dir)
    initial_emotion = validate_emotion_or_exit(args.emotion, artifacts)

    state = ChatState(active_emotion=initial_emotion, strength=args.strength)

    print(f"Loading model {args.model} ...")
    model = InterpretableModel(args.model)
    stop_token_ids = collect_stop_token_ids(model)

    summary_model = artifacts.summary.get("model")
    if summary_model and summary_model != args.model:
        print(f"Note: vectors were produced with {summary_model}, but chat is using {args.model}.")

    print_banner(args.model, artifacts, state, args)

    while True:
        try:
            user_input = input("you> ")
        except EOFError:
            print()
            print("Exiting steered chat.")
            break
        except KeyboardInterrupt:
            print()
            print("Exiting steered chat.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(user_input, args.model, artifacts, state, args)
            if not should_continue:
                print("Exiting steered chat.")
                break
            print()
            continue

        state.messages.append({"role": "user", "content": user_input})
        try:
            assistant_reply = stream_reply(
                model=model,
                artifacts=artifacts,
                state=state,
                args=args,
                stop_token_ids=stop_token_ids,
            )
        except KeyboardInterrupt:
            state.messages.pop()
            print()
            print("Generation interrupted. User turn was not added to history.")
            print()
            continue

        state.messages.append({"role": "assistant", "content": assistant_reply})
        print()


if __name__ == "__main__":
    main()
