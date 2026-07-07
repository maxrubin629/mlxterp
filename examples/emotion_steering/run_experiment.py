#!/usr/bin/env python3
"""
Small-scale replication of the emotion-concepts apparatus for chat models.

Works with any mlx-lm or mlx-vlm chat model (Gemma 4, Qwen 3.5/3.6, Llama, ...).

This script:
1. Generates labeled one-paragraph stories for several emotions.
2. Extracts a residual-stream proxy from a chosen layer with mlxterp.
3. Builds emotion vectors by averaging from token 50 onward and subtracting
   the across-emotion mean.
4. Projects out neutral principal components.
5. Evaluates the vectors on held-out stories.
6. Demonstrates steering with the learned vectors.

All outputs stay inside examples/emotion_steering/artifacts/.

Usage:
    uv run python examples/emotion_steering/run_experiment.py
    uv run python examples/emotion_steering/run_experiment.py -m mlx-community/Qwen3.5-27B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import mlx.core as mx
import numpy as np

from mlxterp import InterpretableModel
from mlxterp import interventions as iv
from mlxterp.core import get_primary_tensor

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_ARTIFACT_BASENAME = "matched_50_story"
DEFAULT_EMOTIONS = ["happy", "sad", "angry", "anxious", "calm"]
EMOTION_GUIDANCE = {
    "happy": (
        "The character should feel light, open, and warmly energized. "
        "Show buoyancy, relief, generous interpretations, and a tendency to notice "
        "color, possibility, or connection."
    ),
    "sad": (
        "The character should feel heavy, reduced, and quietly hurting. "
        "Show slowed movement, lowered energy, attention to absence or loss, and "
        "difficulty taking the next step."
    ),
    "angry": (
        "The character should feel wronged, obstructed, or disrespected. "
        "Show heat, tension, clipped focus, blame, and an urge to confront, correct, or push back."
    ),
    "anxious": (
        "The character should feel watchful and uncertain. "
        "Show rehearsal, scanning, bodily vigilance, imagined mistakes, and a mind that keeps leaping ahead."
    ),
    "calm": (
        "The character should feel settled, regulated, and steady. "
        "Show measured pace, spacious attention, even breathing, and an ability to absorb ambiguity without rushing."
    ),
}
TRAIN_TOPICS = [
    "A few minutes before midnight, the character waits for a phone call that could change where they will live next month.",
    "On the way home, the character opens an email with the final decision on an application they have cared about for years.",
    "The character unlocks the apartment door and finds the kitchen light on even though they expected the place to be empty.",
    "At a community center, the character sees a crowd gathered around a bulletin board where an important list has just been posted.",
    "The character receives a package containing a notebook that belonged to someone they have not seen since childhood.",
    "After a long silence, the character spots their sibling waiting alone on a bench outside a hospital.",
    "A store clerk stops the character near the exit and says there is a problem with the receipt.",
    "The character pulls into the driveway and notices a strange car parked in front of the house.",
    "At work, the character is asked to step into a conference room where two managers are already seated.",
    "The character reaches the train platform just as an announcement changes the destination of the next arrival.",
]
EVAL_TOPICS = [
    "The character sits in a parked car outside a courthouse while rereading a short text message.",
    "When the character opens the mailbox, there is a single handwritten envelope with no return address.",
    "The character hears footsteps in the hallway and then a knock at the door just as dinner is nearly ready.",
    "At dawn, the character checks an online portal and sees that a long-awaited result is finally available.",
]
STEERING_PROMPTS = [
    "Continue in 2-3 sentences: Mina held the envelope in both hands before sliding a finger under the seal.",
    "Continue in 2-3 sentences: Theo paused in the kitchen when he noticed the back door was already open.",
    "Continue in 2-3 sentences: Lena refreshed the results page and watched a new line of text appear.",
]
NEUTRAL_TEXTS = [
    "The instruction manual explains how to assemble the shelf. First, lay the boards on a flat surface and sort the screws by length. Attach the side panels to the base, tighten the fasteners, and add the remaining crossbar. After the frame is secure, place the shelves into the slots and check that the unit stands level on the floor.",
    "A commuter train leaves the station every twenty minutes. The first stop is downtown, followed by the university, the public library, and the riverfront terminal. Riders enter through the front doors, validate their passes, and wait for the announcement before exiting at each platform. Service increases during the evening rush hour and returns to the regular schedule after nine o'clock.",
    "The weather bulletin lists a gradual shift in cloud cover across the region. Light winds move in from the coast during the morning, and temperatures stay mild through the afternoon. Forecasters expect scattered rain in the hills, while lower areas remain mostly dry. Road conditions are normal, and no travel advisories are currently in effect.",
    "The museum catalog describes a ceramic bowl from the late nineteenth century. Its glaze was applied in several thin layers, creating a muted green surface with darker flecks near the rim. The bowl was restored in the 1980s, documented, and later placed in a glass case beside related tools from the same workshop. A short label lists the maker, materials, and estimated date.",
    "The town council agenda begins with road maintenance updates, followed by a review of water usage data and a budget vote on park lighting. Staff members provide summaries of each item, answer questions, and record amendments before the final tally. The meeting closes with notices about the farmers market schedule and next month's public hearing.",
    "A recipe card for vegetable soup lists onions, carrots, celery, tomatoes, beans, stock, and dried herbs. The vegetables are chopped into similar sizes, simmered in a large pot, and stirred occasionally while the beans soften. After forty minutes, the soup is seasoned with salt and pepper, cooled slightly, and stored in containers for later meals.",
    "The field report documents bird activity near a freshwater marsh. Observers arrived at sunrise, measured wind speed, and marked the location of nests along the reeds. Most movement occurred near the shallow northern edge, where several species searched for insects and small fish. The report ends with counts, timestamps, and notes about water level changes.",
    "The software release notes describe a small update to the calendar application. It improves search speed, fixes an issue with recurring reminders, and adjusts the spacing in the weekly view. Users can install the update from the settings menu, restart the app, and confirm the new version number on the about screen.",
    "The grocery inventory sheet tracks rice, pasta, canned beans, cleaning supplies, soap, and paper towels. Quantities are counted every Friday and compared with the previous week's totals. Items below the restock threshold are added to a purchase list, grouped by aisle, and reviewed before the store order is submitted.",
    "A biology worksheet explains how leaves exchange gases through small pores called stomata. When these pores open, carbon dioxide enters and oxygen exits as part of photosynthesis. Water can also leave through the same openings, so the plant balances gas exchange with moisture retention. The worksheet ends with labeled diagrams and short review questions.",
]


@dataclass
class StoryRecord:
    split: str
    emotion: str
    topic: str
    prompt: str
    text: str
    token_count: int


def model_slug(model_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in model_name).strip("_")


def default_output_dir_for_model(model_name: str) -> Path:
    if model_name == MODEL_NAME:
        return SCRIPT_DIR / "artifacts" / DEFAULT_ARTIFACT_BASENAME

    return SCRIPT_DIR / "artifacts" / f"{model_slug(model_name)}_{DEFAULT_ARTIFACT_BASENAME}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for all generated artifacts. Defaults to a model-specific example path.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=MODEL_NAME,
        help="MLX model repo to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for story generation. Gemma-4 works best here at 1.0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Top-k sampling for story generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling for story generation.",
    )
    parser.add_argument(
        "--max-story-tokens",
        type=int,
        default=260,
        help="Max new tokens for story generation.",
    )
    parser.add_argument(
        "--min-story-tokens",
        type=int,
        default=120,
        help="Minimum token count before accepting a generated story.",
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=50,
        help="Token position where averaging begins.",
    )
    parser.add_argument(
        "--neutral-variance-threshold",
        type=float,
        default=0.5,
        help="Explained variance target for neutral PCA projection.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate training and evaluation stories even if cached files exist.",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=8.0,
        help="Multiplier applied to unit-normalized emotion vectors during steering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir_for_model(args.model)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(DEFAULT_EMOTIONS) * len(TRAIN_TOPICS) != 50:
        raise ValueError("Expected exactly 50 training stories from emotions x train topics.")

    print(f"Loading {args.model} ...")
    model = InterpretableModel(args.model)
    layer_idx = max(0, min(len(model.layers) - 1, (2 * len(model.layers)) // 3))
    print(f"Using layer {layer_idx} out of {len(model.layers)} total layers")

    train_path = output_dir / "train_stories.json"
    eval_path = output_dir / "eval_stories.json"
    neutral_path = output_dir / "neutral_texts.json"

    if args.force_regenerate or not train_path.exists():
        train_records = generate_story_split(
            model=model,
            split="train",
            emotions=DEFAULT_EMOTIONS,
            topics=TRAIN_TOPICS,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_story_tokens,
            min_story_tokens=args.min_story_tokens,
        )
        save_story_records(train_path, train_records)
    else:
        train_records = load_story_records(train_path)

    if args.force_regenerate or not eval_path.exists():
        eval_records = generate_story_split(
            model=model,
            split="eval",
            emotions=DEFAULT_EMOTIONS,
            topics=EVAL_TOPICS,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_story_tokens,
            min_story_tokens=args.min_story_tokens,
        )
        save_story_records(eval_path, eval_records)
    else:
        eval_records = load_story_records(eval_path)

    neutral_path.write_text(json.dumps(NEUTRAL_TEXTS, indent=2), encoding="utf-8")

    print("Extracting training activations ...")
    train_means = extract_story_means(model, train_records, layer_idx, args.start_token)
    print("Extracting evaluation activations ...")
    eval_means = extract_story_means(model, eval_records, layer_idx, args.start_token)
    print("Extracting neutral activations ...")
    neutral_means = extract_neutral_means(model, NEUTRAL_TEXTS, layer_idx, args.start_token)

    emotion_vectors = build_emotion_vectors(
        train_means,
        neutral_means,
        variance_threshold=args.neutral_variance_threshold,
    )

    np.savez(
        output_dir / "emotion_vectors.npz",
        global_mean=emotion_vectors["global_mean"],
        neutral_basis=emotion_vectors["neutral_basis"],
        emotions=np.array(DEFAULT_EMOTIONS, dtype=object),
        **{f"vector_{emotion}": emotion_vectors["vectors"][emotion] for emotion in DEFAULT_EMOTIONS},
        **{f"unit_vector_{emotion}": emotion_vectors["unit_vectors"][emotion] for emotion in DEFAULT_EMOTIONS},
    )

    print("Evaluating held-out stories ...")
    evaluation = evaluate_vectors(eval_means, emotion_vectors)

    print("Reading top tokens through the unembed ...")
    token_report = build_token_report(model, emotion_vectors["unit_vectors"])

    print("Running steering demos ...")
    steering_report = build_steering_report(
        model=model,
        unit_vectors=emotion_vectors["unit_vectors"],
        layer_idx=layer_idx,
        strength=args.steering_strength,
    )

    summary = {
        "model": args.model,
        "layer_idx": layer_idx,
        "num_layers": len(model.layers),
        "start_token": args.start_token,
        "emotions": DEFAULT_EMOTIONS,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": None,
            "repetition_penalty": None,
            "presence_penalty": None,
        },
        "dataset_design": {
            "train_topic_count": len(TRAIN_TOPICS),
            "eval_topic_count": len(EVAL_TOPICS),
            "matched_train_stories": len(DEFAULT_EMOTIONS) * len(TRAIN_TOPICS),
            "emotion_guidance": EMOTION_GUIDANCE,
        },
        "train_story_count": len(train_records),
        "eval_story_count": len(eval_records),
        "neutral_text_count": len(NEUTRAL_TEXTS),
        "neutral_pc_count": int(emotion_vectors["neutral_basis"].shape[1]),
        "heldout_accuracy": evaluation["accuracy"],
        "heldout_predictions": evaluation["predictions"],
        "confusion": evaluation["confusion"],
        "vector_norms": {
            emotion: float(np.linalg.norm(emotion_vectors["vectors"][emotion]))
            for emotion in DEFAULT_EMOTIONS
        },
        "top_tokens": token_report,
        "steering": steering_report,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "emotions": DEFAULT_EMOTIONS,
                "sampling": summary["sampling"],
                "train_topics": TRAIN_TOPICS,
                "eval_topics": EVAL_TOPICS,
                "emotion_guidance": EMOTION_GUIDANCE,
                "start_token": args.start_token,
                "neutral_variance_threshold": args.neutral_variance_threshold,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, train_records, eval_records),
        encoding="utf-8",
    )

    print()
    print(f"Held-out top-1 accuracy: {evaluation['accuracy']:.3f}")
    print(f"Neutral PCs projected out: {emotion_vectors['neutral_basis'].shape[1]}")
    print(f"Artifacts written to: {output_dir}")


def save_story_records(path: Path, records: Sequence[StoryRecord]) -> None:
    serializable = [record.__dict__ for record in records]
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def load_story_records(path: Path) -> List[StoryRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [StoryRecord(**record) for record in raw]


def has_chat_template(template_owner: object | None) -> bool:
    return bool(
        template_owner is not None
        and hasattr(template_owner, "apply_chat_template")
        and getattr(template_owner, "chat_template", None)
    )


def chat_prompt(tokenizer, user_text: str, processor=None) -> str:
    template_owner = processor if has_chat_template(processor) else tokenizer
    if has_chat_template(template_owner):
        return template_owner.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {user_text}\nAssistant:"


def generate_text(
    model: InterpretableModel,
    prompt_text: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
) -> str:
    """Generate a completion with mlx-vlm or mlx-lm, depending on how the model loaded."""
    if getattr(model, "processor", None) is not None:
        from mlx_vlm import generate as vlm_generate

        result = vlm_generate(
            model.model,
            model.processor,
            prompt=prompt_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return result.text

    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    return lm_generate(
        model.model,
        model.tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p, top_k=top_k),
    )


def build_story_request(emotion: str, topic: str, split: str) -> str:
    mode_hint = (
        "This is a training example for representation extraction."
        if split == "train"
        else "This is a held-out evaluation example."
    )
    guidance = EMOTION_GUIDANCE[emotion]
    return textwrap.dedent(
        f"""
        Write a single-paragraph story of about 180 to 240 words.
        Situation: {topic}
        The central character should clearly experience the emotion "{emotion}".
        {guidance}
        Use ordinary concrete details instead of abstract analysis.
        Make the emotion legible through perception, pacing, choices, body language, and interpretation.
        Do not use the exact word "{emotion}" in the story.
        Do not directly diagnose the feeling with lines like "they felt {emotion}".
        Keep the prose natural, vivid, and specific.
        Use one named protagonist and a consistent point of view.
        Avoid bullet points, titles, lists, and meta commentary.
        {mode_hint}
        """
    ).strip()


def generate_story_split(
    model: InterpretableModel,
    split: str,
    emotions: Sequence[str],
    topics: Sequence[str],
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    min_story_tokens: int,
) -> List[StoryRecord]:
    records: List[StoryRecord] = []
    total = len(emotions) * len(topics)
    current = 0

    for emotion in emotions:
        for topic in topics:
            current += 1
            prompt = build_story_request(emotion, topic, split)
            story_text = ""
            token_count = 0

            for _attempt in range(3):
                result_text = generate_text(
                    model,
                    prompt_text=chat_prompt(model.tokenizer, prompt, processor=model.processor),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                story_text = clean_generation_text(result_text)
                token_count = len(model.encode(story_text))
                if token_count >= min_story_tokens:
                    break

            print(f"[{current:02d}/{total}] {split} {emotion:>7} | {topic} | {token_count} tokens")
            records.append(
                StoryRecord(
                    split=split,
                    emotion=emotion,
                    topic=topic,
                    prompt=prompt,
                    text=story_text,
                    token_count=token_count,
                )
            )

    return records


def clean_generation_text(text: str) -> str:
    return " ".join(text.strip().split())


def extract_story_means(
    model: InterpretableModel,
    records: Sequence[StoryRecord],
    layer_idx: int,
    start_token: int,
) -> List[Dict[str, object]]:
    outputs: List[Dict[str, object]] = []
    for record in records:
        layer_output = layer_mean_for_text(model, record.text, layer_idx, start_token)
        outputs.append(
            {
                "split": record.split,
                "emotion": record.emotion,
                "topic": record.topic,
                "text": record.text,
                "token_count": record.token_count,
                "mean": layer_output["mean"],
                "effective_start": layer_output["effective_start"],
                "seq_len": layer_output["seq_len"],
            }
        )
    return outputs


def extract_neutral_means(
    model: InterpretableModel,
    texts: Sequence[str],
    layer_idx: int,
    start_token: int,
) -> np.ndarray:
    means = []
    for text in texts:
        layer_output = layer_mean_for_text(model, text, layer_idx, start_token)
        means.append(layer_output["mean"])
    return np.stack(means, axis=0)


def mlx_to_numpy(array: mx.array, dtype: np.dtype = np.float32) -> np.ndarray:
    # bfloat16 has no numpy equivalent, so cast before conversion
    return np.array(array.astype(mx.float32)).astype(dtype, copy=False)


def layer_mean_for_text(
    model: InterpretableModel,
    text: str,
    layer_idx: int,
    start_token: int,
) -> Dict[str, object]:
    trace_input = chat_prompt(model.tokenizer, text, processor=model.processor)
    with model.trace(trace_input):
        layer_output = model.layers[layer_idx].output.save()

    sequence = mlx_to_numpy(get_primary_tensor(layer_output))[0]
    seq_len = int(sequence.shape[0])
    effective_start = min(start_token, max(0, seq_len - 1))
    mean = sequence[effective_start:].mean(axis=0).astype(np.float32)
    return {
        "mean": mean,
        "effective_start": effective_start,
        "seq_len": seq_len,
    }


def build_emotion_vectors(
    train_means: Sequence[Dict[str, object]],
    neutral_means: np.ndarray,
    variance_threshold: float,
) -> Dict[str, object]:
    grouped: Dict[str, List[np.ndarray]] = {emotion: [] for emotion in DEFAULT_EMOTIONS}
    for item in train_means:
        grouped[item["emotion"]].append(item["mean"])

    emotion_centroids = {
        emotion: np.stack(grouped[emotion], axis=0).mean(axis=0)
        for emotion in DEFAULT_EMOTIONS
    }
    global_mean = np.stack(list(emotion_centroids.values()), axis=0).mean(axis=0)
    neutral_basis = compute_neutral_basis(neutral_means, variance_threshold)

    vectors: Dict[str, np.ndarray] = {}
    unit_vectors: Dict[str, np.ndarray] = {}
    for emotion, centroid in emotion_centroids.items():
        raw = centroid - global_mean
        projected = project_out(raw, neutral_basis).astype(np.float32)
        norm = float(np.linalg.norm(projected))
        if norm == 0.0:
            unit = projected
        else:
            unit = (projected / norm).astype(np.float32)
        vectors[emotion] = projected
        unit_vectors[emotion] = unit

    return {
        "emotion_centroids": emotion_centroids,
        "global_mean": global_mean.astype(np.float32),
        "neutral_basis": neutral_basis.astype(np.float32),
        "vectors": vectors,
        "unit_vectors": unit_vectors,
    }


def compute_neutral_basis(neutral_means: np.ndarray, variance_threshold: float) -> np.ndarray:
    centered = neutral_means - neutral_means.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[1], 0), dtype=np.float32)

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values ** 2
    total = float(variances.sum())
    if total <= 0.0:
        return np.zeros((centered.shape[1], 0), dtype=np.float32)

    explained = np.cumsum(variances / total)
    k = int(np.searchsorted(explained, variance_threshold, side="left") + 1)
    k = max(1, min(k, vt.shape[0]))
    return vt[:k].T


def project_out(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return vector
    return vector - basis @ (basis.T @ vector)


def evaluate_vectors(
    eval_means: Sequence[Dict[str, object]],
    emotion_vectors: Dict[str, object],
) -> Dict[str, object]:
    global_mean = emotion_vectors["global_mean"]
    unit_vectors = emotion_vectors["unit_vectors"]
    predictions = []
    confusion: Dict[str, Dict[str, int]] = {
        emotion: dict.fromkeys(DEFAULT_EMOTIONS, 0)
        for emotion in DEFAULT_EMOTIONS
    }

    correct = 0
    for item in eval_means:
        centered = item["mean"] - global_mean
        scores = {
            emotion: float(np.dot(centered, unit_vectors[emotion]))
            for emotion in DEFAULT_EMOTIONS
        }
        predicted = max(scores, key=scores.get)
        if predicted == item["emotion"]:
            correct += 1
        confusion[item["emotion"]][predicted] += 1
        predictions.append(
            {
                "topic": item["topic"],
                "target_emotion": item["emotion"],
                "predicted_emotion": predicted,
                "scores": scores,
            }
        )

    accuracy = correct / max(1, len(eval_means))
    return {"accuracy": accuracy, "predictions": predictions, "confusion": confusion}


def build_token_report(
    model: InterpretableModel,
    unit_vectors: Dict[str, np.ndarray],
    top_k: int = 8,
) -> Dict[str, Dict[str, List[str]]]:
    report: Dict[str, Dict[str, List[str]]] = {}
    final_norm = model._module_resolver.get_final_norm()

    for emotion, vector in unit_vectors.items():
        hidden = mx.array(vector)
        if final_norm is not None:
            hidden = final_norm(hidden)

        up = model.get_token_predictions(hidden, top_k=top_k, return_scores=True)
        down = model.get_token_predictions(-hidden, top_k=top_k, return_scores=True)
        report[emotion] = {
            "upweighted": [sanitize_token(model.token_to_str(token_id)) for token_id, _ in up],
            "downweighted": [sanitize_token(model.token_to_str(token_id)) for token_id, _ in down],
        }

    return report


def sanitize_token(token: str) -> str:
    return token.replace("\n", "\\n").strip() or "<whitespace>"


def build_steering_report(
    model: InterpretableModel,
    unit_vectors: Dict[str, np.ndarray],
    layer_idx: int,
    strength: float,
) -> List[Dict[str, object]]:
    reports = []
    intervention_key = f"layers.{layer_idx}"
    probe_tokens = build_probe_tokens(model)

    for prompt in STEERING_PROMPTS:
        prompt_text = chat_prompt(model.tokenizer, prompt, processor=model.processor)

        with model.trace(prompt_text):
            baseline_logits = model.output.save()
        mx.eval(baseline_logits)

        baseline_scores = collect_probe_scores(baseline_logits, probe_tokens)
        steered = {}
        for emotion in DEFAULT_EMOTIONS:
            with model.trace(
                prompt_text,
                interventions={
                    intervention_key: iv.add_vector(mx.array(unit_vectors[emotion] * strength))
                },
            ):
                steered_logits = model.output.save()
            mx.eval(steered_logits)

            steered_scores = collect_probe_scores(steered_logits, probe_tokens)
            deltas = {
                target: steered_scores[target] - baseline_scores[target]
                for target in DEFAULT_EMOTIONS
            }
            steered[emotion] = {
                "steered_scores": steered_scores,
                "delta_vs_baseline": deltas,
            }

        reports.append(
            {
                "prompt": prompt,
                "probe_tokens": probe_tokens,
                "baseline_scores": baseline_scores,
                "steered": steered,
            }
        )

    return reports


def build_probe_tokens(model: InterpretableModel) -> Dict[str, Dict[str, object]]:
    tokens = {}
    for emotion in DEFAULT_EMOTIONS:
        encoded = model.encode(f" {emotion}")
        token_id = int(encoded[-1])
        tokens[emotion] = {
            "token_id": token_id,
            "token_text": sanitize_token(model.token_to_str(token_id)),
            "token_count": len(encoded),
        }
    return tokens


def collect_probe_scores(
    logits: mx.array,
    probe_tokens: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    return {
        emotion: float(logits[0, -1, probe_info["token_id"]])
        for emotion, probe_info in probe_tokens.items()
    }


def build_report(
    summary: Dict[str, object],
    train_records: Sequence[StoryRecord],
    eval_records: Sequence[StoryRecord],
) -> str:
    lines = [
        "# Emotion Concepts Report",
        "",
        f"- Model: `{summary['model']}`",
        f"- Layer: `{summary['layer_idx']}` / `{summary['num_layers'] - 1}`",
        f"- Emotions: {', '.join(summary['emotions'])}",
        f"- Train stories: `{summary['train_story_count']}`",
        f"- Eval stories: `{summary['eval_story_count']}`",
        f"- Neutral texts: `{summary['neutral_text_count']}`",
        (
            "- Sampling: "
            f"`temperature={summary['sampling']['temperature']}`, "
            f"`top_k={summary['sampling']['top_k']}`, "
            f"`top_p={summary['sampling']['top_p']}`, "
            "`min_p=None`, `repetition_penalty=None`, `presence_penalty=None`"
        ),
        f"- Neutral PCs projected out: `{summary['neutral_pc_count']}`",
        f"- Held-out top-1 accuracy: `{summary['heldout_accuracy']:.3f}`",
        "",
        "## Dataset Design",
        "",
        (
            f"- Matched train situations: `{summary['dataset_design']['train_topic_count']}` "
            f"x `{len(summary['emotions'])}` emotions = `{summary['dataset_design']['matched_train_stories']}` stories"
        ),
        f"- Held-out evaluation situations: `{summary['dataset_design']['eval_topic_count']}`",
        "",
        "## Train Story Counts",
        "",
    ]

    for emotion in DEFAULT_EMOTIONS:
        count = sum(1 for record in train_records if record.emotion == emotion)
        avg_tokens = np.mean([record.token_count for record in train_records if record.emotion == emotion])
        lines.append(f"- `{emotion}`: {count} stories, {avg_tokens:.1f} tokens on average")

    lines.extend(
        [
            "",
            "## Held-Out Confusion",
            "",
        ]
    )
    for emotion in DEFAULT_EMOTIONS:
        row = summary["confusion"][emotion]
        row_text = ", ".join(f"{candidate}:{row[candidate]}" for candidate in DEFAULT_EMOTIONS)
        lines.append(f"- `{emotion}` -> {row_text}")

    lines.extend(
        [
            "",
            "## Logit-Lens Tokens",
            "",
        ]
    )
    for emotion in DEFAULT_EMOTIONS:
        token_info = summary["top_tokens"][emotion]
        up = ", ".join(token_info["upweighted"])
        down = ", ".join(token_info["downweighted"])
        lines.append(f"- `{emotion}` up: {up}")
        lines.append(f"- `{emotion}` down: {down}")

    lines.extend(
        [
            "",
            "## Steering Effects",
            "",
        ]
    )
    for item in summary["steering"]:
        lines.append("### Prompt")
        lines.append("")
        lines.append(item["prompt"])
        lines.append("")
        baseline = ", ".join(
            f"{emotion}:{score:.2f}"
            for emotion, score in item["baseline_scores"].items()
        )
        lines.append(f"- Baseline probe logits: {baseline}")
        for emotion in DEFAULT_EMOTIONS:
            deltas = item["steered"][emotion]["delta_vs_baseline"]
            delta_text = ", ".join(
                f"{target}:{deltas[target]:+.2f}"
                for target in DEFAULT_EMOTIONS
            )
            lines.append(f"- {emotion} intervention deltas: {delta_text}")
        lines.append("")

    lines.extend(
        [
            "## Held-Out Story Topics",
            "",
        ]
    )
    for record in eval_records:
        lines.append(f"- `{record.emotion}` on `{record.topic}`")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
