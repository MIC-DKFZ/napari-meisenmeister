from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from meisenmeister.training.predict import predict_case_from_files


class MeisenMeisterInferenceError(RuntimeError):
    pass


DEFAULT_MODEL_REPO_ID = "Bubenpo/MeisenMeister"
DEFAULT_MODEL_REVISION = "v1"
SOURCE_METADATA_KEYS = (
    "path",
    "source_path",
    "file_path",
    "filename",
    "filepath",
)


def validate_model_folder(model_folder: str | Path) -> Path:
    path = Path(model_folder).expanduser().resolve()
    required_paths = [
        path / "dataset.json",
        path / "mmPlans.json",
        path / "fold_all",
        path / "fold_all" / "model_best.pt",
    ]
    missing = [required for required in required_paths if not required.exists()]
    if missing:
        formatted = ", ".join(str(item) for item in missing)
        raise MeisenMeisterInferenceError(
            f"Model folder is missing required MeisenMeister files: {formatted}"
        )
    return path


def resolve_model_folder(model_folder: str | Path | None) -> Path:
    if model_folder and str(model_folder).strip():
        return validate_model_folder(model_folder)

    snapshot_path = snapshot_download(
        repo_id=DEFAULT_MODEL_REPO_ID,
        revision=DEFAULT_MODEL_REVISION,
        allow_patterns=[
            "dataset.json",
            "mmPlans.json",
            "fold_all/*",
        ],
    )
    return validate_model_folder(snapshot_path)


def initialize_inference_runtime(model_folder: str | Path | None) -> dict[str, Any]:
    model_path = resolve_model_folder(model_folder)
    return {
        "model_folder": str(model_path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_candidate_path(candidate: Any) -> Path | None:
    if candidate is None:
        return None
    try:
        path = Path(str(candidate)).expanduser()
    except Exception:
        return None
    if not path.is_file():
        return None
    return path.resolve()


def resolve_layer_source_path(layer: Any) -> Path:
    source = getattr(layer, "source", None)
    source_path = _normalize_candidate_path(getattr(source, "path", None))
    if source_path is not None:
        return source_path

    metadata = getattr(layer, "metadata", {}) or {}
    for key in SOURCE_METADATA_KEYS:
        source_path = _normalize_candidate_path(metadata.get(key))
        if source_path is not None:
            return source_path

    layer_name = getattr(layer, "name", "<unnamed layer>")
    raise MeisenMeisterInferenceError(
        "MeisenMeister classification requires original NIfTI-backed layers. "
        f"Could not resolve a source file for layer '{layer_name}'. "
        "Please open the original image files in napari and select those layers."
    )


def resolve_case_source_files(
    layers_by_channel: dict[str, Any],
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    required_channels = ("pre", "post1", "post2")
    missing_channels = [
        channel_name for channel_name in required_channels if channel_name not in layers_by_channel
    ]
    if missing_channels:
        formatted = ", ".join(missing_channels)
        raise MeisenMeisterInferenceError(f"Missing required channels: {formatted}")

    resolved_paths: dict[str, Path] = {}
    diagnostics: list[dict[str, str]] = []
    for channel_name in required_channels:
        layer = layers_by_channel[channel_name]
        source_path = resolve_layer_source_path(layer)
        resolved_paths[channel_name] = source_path
        diagnostics.append(
            {
                "channel": channel_name,
                "layer_name": str(getattr(layer, "name", channel_name)),
                "source_path": str(source_path),
            }
        )

    distinct_paths = {str(path) for path in resolved_paths.values()}
    if len(distinct_paths) != len(required_channels):
        raise MeisenMeisterInferenceError(
            "MeisenMeister classification requires three distinct source files for "
            "pre, post1, and post2."
        )
    return resolved_paths, diagnostics


def classify_case_from_layers(
    *,
    model_folder: str | Path | None,
    layers_by_channel: dict[str, Any],
) -> dict[str, Any]:
    model_path = resolve_model_folder(model_folder)
    plans = _load_json(model_path / "mmPlans.json")
    source_files_by_channel, diagnostics = resolve_case_source_files(layers_by_channel)
    input_diagnostics = list(diagnostics)

    try:
        prediction_result = predict_case_from_files(
            str(model_path),
            str(source_files_by_channel["pre"]),
            str(source_files_by_channel["post1"]),
            str(source_files_by_channel["post2"]),
            folds=["all"],
        )
    except Exception as exc:
        raise MeisenMeisterInferenceError(str(exc)) from exc

    roi_labels = plans.get("roi_labels", {}) or {}
    label_map = {}
    for side_name in ("left", "right"):
        if side_name in roi_labels:
            try:
                label_map[side_name] = int(roi_labels[side_name])
            except (TypeError, ValueError):
                continue

    return {
        "model_folder": str(model_path),
        "run_directory": prediction_result.get("run_directory"),
        "predictions_path": prediction_result.get("predictions_path"),
        "concise_output_path": None,
        "probabilities": prediction_result["concise_predictions"],
        "predictions_payload": None,
        "label_map": label_map,
        "mask": {
            "path": prediction_result["mask_path"],
            "source_path": str(source_files_by_channel["pre"]),
        },
        "input_diagnostics": input_diagnostics,
        "staging_details": [
            {
                "channel": channel_name,
                "source_path": str(source_files_by_channel[channel_name]),
                "staging_method": "predict_case_from_files",
            }
            for channel_name in ("pre", "post1", "post2")
        ],
    }
