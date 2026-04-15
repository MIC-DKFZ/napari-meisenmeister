from __future__ import annotations

from typing import Any, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _pretty_class_name(label_name: str) -> str:
    return label_name.replace("_", " ").strip().title()


SIDE_COLORS = {
    "left": "#4a7bb7",
    "right": "#d37a45",
}


def side_color(side_name: str) -> str:
    return SIDE_COLORS.get(side_name.lower(), "#4a7bb7")


class ProbabilityCard(QFrame):
    def __init__(
        self, side_name: str, probabilities: dict[str, float], parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setObjectName("ProbabilityCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        accent_color = side_color(side_name)
        self.setStyleSheet(
            """
            QFrame#ProbabilityCard {
                border: 1px solid #d7dbe2;
                border-radius: 10px;
                background: #ffffff;
            }
            QLabel[classRole="eyebrow"] {
                color: #5f6b7a;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel[classRole="headline"] {
                color: #18212b;
                font-size: 17px;
                font-weight: 700;
            }
            QLabel[classRole="winner"] {
                color: """
            + accent_color
            + """;
                font-weight: 600;
            }
            QLabel[classRole="classLabel"] {
                color: #243141;
                font-size: 12px;
                font-weight: 500;
            }
            QLabel[classRole="classValue"] {
                color: #18212b;
                font-size: 12px;
                font-weight: 600;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.setLayout(layout)

        pretty_side = f"{side_name.capitalize()} breast"
        winner = max(probabilities, key=probabilities.get)

        eyebrow = QLabel(pretty_side)
        eyebrow.setProperty("classRole", "eyebrow")
        layout.addWidget(eyebrow)

        headline_row = QHBoxLayout()
        headline = QLabel(_pretty_class_name(winner))
        headline.setProperty("classRole", "headline")
        headline_row.addWidget(headline)
        headline_row.addStretch()

        chip = QLabel(f"{probabilities[winner] * 100:.1f}%")
        chip.setProperty("classRole", "winner")
        headline_row.addWidget(chip)
        layout.addLayout(headline_row)

        for label_name, value in probabilities.items():
            row = QVBoxLayout()
            row.setSpacing(3)

            row_header = QHBoxLayout()
            row_label = QLabel(_pretty_class_name(label_name))
            row_label.setProperty("classRole", "classLabel")
            row_header.addWidget(row_label)
            row_header.addStretch()

            row_value = QLabel(f"{value * 100:.1f}%")
            row_value.setProperty("classRole", "classValue")
            row_header.addWidget(row_value)

            progress = QProgressBar()
            progress.setRange(0, 1000)
            progress.setValue(int(round(value * 1000)))
            progress.setTextVisible(False)
            progress.setStyleSheet(
                """
                QProgressBar {
                    border: none;
                    border-radius: 4px;
                    background: #e8edf3;
                    min-height: 8px;
                }
                QProgressBar::chunk {
                    background: """
                + accent_color
                + """;
                    border-radius: 4px;
                }
                """
            )
            row.addLayout(row_header)
            row.addWidget(progress)
            layout.addLayout(row)


class MeisenMeisterGUI(QWidget):
    CHANNEL_NAMES = ("pre", "post1", "post2")

    def __init__(
        self,
        viewer: Any = None,
        *,
        napari_viewer: Any = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._viewer = napari_viewer if napari_viewer is not None else viewer
        if self._viewer is None:
            raise TypeError("MeisenMeisterGUI requires a napari viewer instance")
        self.setMinimumWidth(420)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        self.setLayout(main_layout)

        title = QLabel("MeisenMeister")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1f2a37;")
        main_layout.addWidget(title)

        subtitle = QLabel(
            "Assign pre, post1, and post2, then classify both breasts with MeisenMeister."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #5b6470;")
        main_layout.addWidget(subtitle)

        main_layout.addWidget(self._init_model_selection())
        main_layout.addWidget(self._init_image_selection())
        main_layout.addWidget(self._init_classify_button())
        main_layout.addWidget(self._init_status_label())
        main_layout.addWidget(self._init_results_group())
        main_layout.addStretch()

        self._connect_layer_events()
        self._update_image_layers()
        self._is_initialized = False
        self._set_initialized(False)
        self._update_classify_enabled_state()

    def _init_model_selection(self) -> QGroupBox:
        group_box = QGroupBox("Model Folder")
        layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText(
            "Optional local model folder. Leave empty to auto-download MeisenMeister v1..."
        )
        self.model_path_input.textChanged.connect(self._update_classify_enabled_state)
        path_layout.addWidget(self.model_path_input)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_model_folder)
        path_layout.addWidget(self.browse_button)

        layout.addLayout(path_layout)

        self.init_button = QPushButton("Initialize")
        self.init_button.clicked.connect(self.on_initialize)
        layout.addWidget(self.init_button)

        note = QLabel(
            "Initialize downloads or validates the default v1 model. "
            "You can also choose a folder containing dataset.json, mmPlans.json, "
            "and fold_all/model_best.pt."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(note)

        group_box.setLayout(layout)
        return group_box

    def _browse_model_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select MeisenMeister model folder")
        if selected:
            self.model_path_input.setText(selected)

    def _init_image_selection(self) -> QGroupBox:
        group_box = QGroupBox("Image Assignment")
        layout = QVBoxLayout()
        self.channel_selectors: dict[str, QComboBox] = {}

        for channel_name in self.CHANNEL_NAMES:
            row = QHBoxLayout()
            label = QLabel(channel_name)
            label.setMinimumWidth(48)
            combo = QComboBox()
            combo.currentIndexChanged.connect(self._update_classify_enabled_state)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.channel_selectors[channel_name] = combo
            row.addWidget(label)
            row.addWidget(combo)
            layout.addLayout(row)

        helper = QLabel("Use three distinct volumetric image layers from the viewer.")
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(helper)

        space_note = QLabel(
            "Masks are displayed in source-file space. Napari may render oblique inputs approximately."
        )
        space_note.setWordWrap(True)
        space_note.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(space_note)

        group_box.setLayout(layout)
        return group_box

    def _connect_layer_events(self) -> None:
        layers = getattr(self._viewer, "layers", None)
        if layers is None:
            return
        try:
            layers.events.inserted.connect(self._update_image_layers)
            layers.events.removed.connect(self._update_image_layers)
            layers.events.reordered.connect(self._update_image_layers)
        except Exception:
            return

    def _viewer_image_layers(self) -> list[Any]:
        layers = getattr(self._viewer, "layers", [])
        image_layers = []
        for layer in layers:
            ndim = getattr(getattr(layer, "data", None), "ndim", None)
            if ndim == 3:
                image_layers.append(layer)
        return image_layers

    def _update_image_layers(self, event: Any = None) -> None:
        current_values = {
            name: combo.currentData() for name, combo in self.channel_selectors.items()
        }
        image_layers = self._viewer_image_layers()
        for channel_name, combo in self.channel_selectors.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select layer...", None)
            for layer in image_layers:
                combo.addItem(layer.name, layer.name)
            selected_value = current_values[channel_name]
            if selected_value is not None:
                index = combo.findData(selected_value)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._update_classify_enabled_state()

    @property
    def selected_image_layers(self) -> dict[str, Any]:
        layers = getattr(self._viewer, "layers", [])
        by_name = {layer.name: layer for layer in layers}
        selected = {}
        for channel_name, combo in self.channel_selectors.items():
            layer_name = combo.currentData()
            if layer_name and layer_name in by_name:
                selected[channel_name] = by_name[layer_name]
        return selected

    def _init_classify_button(self) -> QGroupBox:
        group_box = QGroupBox("")
        layout = QVBoxLayout()
        self.classify_button = QPushButton("Classify")
        self.classify_button.clicked.connect(self.on_classify)
        layout.addWidget(self.classify_button)
        group_box.setLayout(layout)
        return group_box

    def _init_status_label(self) -> QWidget:
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "QLabel { color: #7ddb5a; font-weight: 700; font-size: 13px; }"
        )
        return self.status_label

    def _init_results_group(self) -> QGroupBox:
        group_box = QGroupBox("Results")
        outer_layout = QVBoxLayout()

        self.results_placeholder = QLabel(
            "No prediction yet. Run classification to see left and right probability cards."
        )
        self.results_placeholder.setWordWrap(True)
        self.results_placeholder.setStyleSheet("color: #6b7280;")
        outer_layout.addWidget(self.results_placeholder)

        self.results_container = QWidget()
        self.results_layout = QHBoxLayout()
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(12)
        self.results_container.setLayout(self.results_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.results_container)
        outer_layout.addWidget(scroll)

        group_box.setLayout(outer_layout)
        return group_box

    def clear_results(self) -> None:
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.results_placeholder.show()

    def populate_results(self, probability_payload: dict[str, dict[str, float]]) -> None:
        self.clear_results()
        swap = {"left": "right", "right": "left"}
        corrected = {swap.get(k, k): v for k, v in probability_payload.items()}
        self.results_layout.addStretch()
        for side_name, probabilities in sorted(corrected.items()):
            self.results_layout.addWidget(ProbabilityCard(side_name, probabilities))
        self.results_layout.addStretch()
        self.results_placeholder.hide()

    def _set_initialized(self, is_initialized: bool) -> None:
        self._is_initialized = is_initialized
        for combo in self.channel_selectors.values():
            combo.setEnabled(is_initialized)
        self._update_classify_enabled_state()

    def _update_classify_enabled_state(self) -> None:
        if not getattr(self, "_is_initialized", False):
            self.classify_button.setEnabled(False)
            return
        selected_layers = self.selected_image_layers
        distinct_count = len({layer.name for layer in selected_layers.values()})
        enabled = len(selected_layers) == 3 and distinct_count == 3
        self.classify_button.setEnabled(enabled)

    def on_initialize(self) -> None:
        raise NotImplementedError

    def on_classify(self) -> None:
        raise NotImplementedError
