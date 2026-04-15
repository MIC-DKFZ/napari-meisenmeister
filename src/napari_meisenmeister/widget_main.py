from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import SimpleITK as sitk
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import QWidget

from .inference import (
    MeisenMeisterInferenceError,
    classify_case_from_layers,
    initialize_inference_runtime,
    resolve_case_source_files,
    validate_model_folder,
)
from .widget_gui import MeisenMeisterGUI, side_color


def _notify(level: str, message: str) -> None:
    try:
        from napari.utils.notifications import show_error, show_info, show_warning  # type: ignore
    except Exception:
        return

    mapping = {
        "info": show_info,
        "warning": show_warning,
        "error": show_error,
    }
    mapping[level](message)


class InitializationThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, *, model_folder: str) -> None:
        super().__init__()
        self.model_folder = model_folder

    def run(self) -> None:
        try:
            self.finished.emit(initialize_inference_runtime(self.model_folder))
        except Exception as exc:
            self.error.emit(str(exc))


class ProcessingThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        model_folder: str,
        selected_layers: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model_folder = model_folder
        self.selected_layers = selected_layers

    def run(self) -> None:
        try:
            result = classify_case_from_layers(
                model_folder=self.model_folder,
                layers_by_channel=self.selected_layers,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MeisenMeisterWidget(MeisenMeisterGUI):
    def __init__(
        self,
        viewer: Any = None,
        *,
        napari_viewer: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(viewer, napari_viewer=napari_viewer, parent=parent)
        self.initialization_thread: InitializationThread | None = None
        self.processing_thread: ProcessingThread | None = None
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self._update_spinner)
        self.spinner_index = 0
        self.spinner_frames = [".", "..", "...", "..", ".", "..", "...", ".."]
        self._spinner_message = ""
        self._last_run_metadata: dict[str, Any] | None = None
        self._initialized_model_folder: str | None = None
        self._ignore_model_path_change = False
        self.model_path_input.textChanged.connect(self._on_model_path_changed)

    def _update_spinner(self) -> None:
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        dots = self.spinner_frames[self.spinner_index]
        self.status_label.setText(f"{self._spinner_message}{dots}")

    def _start_processing(self, message: str = "Working...") -> None:
        self.init_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        self.model_path_input.setEnabled(False)
        self.browse_button.setEnabled(False)
        for combo in self.channel_selectors.values():
            combo.setEnabled(False)
        self._spinner_message = message
        self.spinner_index = 0
        self.status_label.setText(f"{message}{self.spinner_frames[0]}")
        self.spinner_timer.start(300)

    def _stop_processing(self, message: str) -> None:
        self.spinner_timer.stop()
        self.status_label.setText(message)
        self.init_button.setEnabled(True)
        self.model_path_input.setEnabled(True)
        self.browse_button.setEnabled(True)
        self._set_initialized(self._initialized_model_folder is not None)
        self._update_classify_enabled_state()

    def _on_model_path_changed(self) -> None:
        if self._ignore_model_path_change:
            return
        self._initialized_model_folder = None
        self._set_initialized(False)
        self.clear_results()
        if not self.spinner_timer.isActive():
            self.status_label.setText("")

    def on_initialize(self) -> None:
        model_folder = self.model_path_input.text().strip()
        if model_folder:
            try:
                validate_model_folder(model_folder)
            except MeisenMeisterInferenceError as exc:
                _notify("error", str(exc))
                return

        self.clear_results()
        if model_folder:
            self._start_processing("Initializing model...")
        else:
            self._start_processing("Downloading default v1 model...")

        self.initialization_thread = InitializationThread(model_folder=model_folder)
        self.initialization_thread.finished.connect(self._on_initialization_finished)
        self.initialization_thread.error.connect(self._on_initialization_error)
        self.initialization_thread.start()

    def _on_initialization_finished(self, runtime: dict[str, Any]) -> None:
        self._initialized_model_folder = runtime["model_folder"]
        self._ignore_model_path_change = True
        self.model_path_input.setText(runtime["model_folder"])
        self._ignore_model_path_change = False
        self._set_initialized(True)
        self._stop_processing("Model initialized")
        _notify("info", "MeisenMeister model ready. You can now assign images and classify.")

    def _on_initialization_error(self, error_message: str) -> None:
        self._initialized_model_folder = None
        self._set_initialized(False)
        self._stop_processing("Initialization failed")
        _notify("error", f"MeisenMeister initialization failed: {error_message}")

    def on_classify(self) -> None:
        if self._initialized_model_folder is None:
            _notify("warning", "Please initialize the model first.")
            return

        selected_layers = self.selected_image_layers
        if len(selected_layers) != 3:
            _notify("warning", "Please assign pre, post1, and post2 to three image layers.")
            return

        distinct_names = {layer.name for layer in selected_layers.values()}
        if len(distinct_names) != 3:
            _notify("warning", "Please use three distinct image layers.")
            return
        try:
            resolve_case_source_files(selected_layers)
        except MeisenMeisterInferenceError as exc:
            _notify("error", str(exc))
            return

        self.clear_results()
        self._start_processing("Thinking")
        self.processing_thread = ProcessingThread(
            model_folder=self._initialized_model_folder,
            selected_layers=selected_layers,
        )
        self.processing_thread.finished.connect(self._on_processing_finished)
        self.processing_thread.error.connect(self._on_processing_error)
        self.processing_thread.start()

    def _on_processing_finished(self, result: dict[str, Any]) -> None:
        self._last_run_metadata = result
        self._stop_processing("Done")
        self._add_masks_to_viewer(result)
        self.populate_results(result["probabilities"])
        _notify("info", "Classification finished successfully.")

    def _on_processing_error(self, error_message: str) -> None:
        self._stop_processing("Failed")
        self.clear_results()
        _notify("error", f"MeisenMeister classification failed: {error_message}")

    def _add_masks_to_viewer(self, result: dict[str, Any]) -> None:
        mask_payload = result["mask"]
        loaded_layers = self._viewer.open(
            mask_payload["path"],
            plugin="napari-nifti",
            layer_type="labels",
        )
        if not loaded_layers:
            raise RuntimeError(f"Could not open predicted mask: {mask_payload['path']}")

        layer = loaded_layers[0]
        layer.name = "MeisenMeister side mask"
        layer.opacity = 0.5
        self._apply_mask_side_colors(layer, result.get("label_map", {}))
        layer.metadata = {
            **getattr(layer, "metadata", {}),
            "meisenmeister": {
                "model_folder": result["model_folder"],
                "run_directory": result["run_directory"],
                "predictions_path": result["predictions_path"],
                "concise_output_path": result["concise_output_path"],
                "mask_path": mask_payload["path"],
                "source_path": mask_payload.get("source_path"),
                "label_map": result.get("label_map", {}),
            },
        }

    @staticmethod
    def _apply_mask_side_colors(layer: Any, label_map: dict[str, Any]) -> None:
        color_mapping: dict[Any, str] = {0: "transparent"}
        for side_name, label_value in label_map.items():
            try:
                label_id = int(label_value)
            except (TypeError, ValueError):
                continue
            color_mapping[label_id] = side_color(side_name)

        try:
            layer.color = color_mapping
            refresh = getattr(layer, "refresh", None)
            if callable(refresh):
                refresh()
        except Exception:
            return

    @staticmethod
    def load_probability_payload(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def load_mask_array(path: str | Path) -> Any:
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image)
