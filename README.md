# napari-meisenmeister

![UI](images/UI.png)

`napari-meisenmeister` brings the MeisenMeister bilateral breast DCE-MRI classifier into napari.

MeisenMeister was used for the winning solution of the [MICCAI 2025 ODELIA Breast MRI Challenge](https://odelia2025.grand-challenge.org/). The upstream framework lives at [MIC-DKFZ/MeisenMeister](https://github.com/MIC-DKFZ/MeisenMeister); this plugin wraps its portable inference mode for interactive use in napari.

The widget is built for the MeisenMeister `v1` portable model layout and expects three co-registered 3D volumes per case — one pre-contrast scan and two post-contrast timepoints from a dynamic contrast-enhanced study:

- `pre`
- `post1`
- `post2`

When you click `Classify`, the plugin writes a temporary single-case MeisenMeister input folder, runs portable inference with `fold_all`, creates breast / left / right segmentations under the hood, and renders per-side probability cards for `healthy`, `benign`, and `malignant`.

## Installation

Create and activate a conda environment first:

```bash
conda create -n napari-meisenmeister python=3.12 -y
conda activate napari-meisenmeister
```

Then install everything with:

```bash
git clone https://github.com/MIC-DKFZ/napari-meisenmeister.git
cd napari-meisenmeister
pip install -e .
```

That single editable install pulls in the napari plugin, the MeisenMeister runtime, and the Hugging Face client used for automatic model download.

## Usage

Launch napari with the widget preloaded:

```bash
napari -w napari-meisenmeister
```

Then:

1. Drag the three 3D DCE-MRI volumes into napari.
2. Assign the image layers to `pre`, `post1`, and `post2` in the widget.
3. Click `Classify`.
4. On the first run, the plugin downloads the default MeisenMeister `v1` model automatically.
5. Review the generated label layers and the left/right probability cards in the widget.

If you already have a local model folder, you can still paste or browse to it manually.

## Notes

- This first version assumes bilateral input and expects both `left` and `right` outputs.
- The MeisenMeister pipeline depends on BreastDivider for the under-the-hood breast segmentation step.
- The default model source is `Bubenpo/MeisenMeister` on Hugging Face at revision `v1`.

## License

This repository is licensed under Apache-2.0. MeisenMeister model weights follow the Hugging Face model card license terms (CC BY-NC-SA 4.0).

## Citation

If you use this plugin in research, please cite the MeisenMeister paper:

```
Hamm, B., Kirchhoff, Y., Rokuss, M., and Maier-Hein, K.,
MeisenMeister: A Simple Two Stage Pipeline for Breast Cancer Classification on MRI,
arXiv:2510.27326 [cs.CV], 2025.
```

```bibtex
@article{hamm2025meisenmeister,
  title={MeisenMeister: A Simple Two Stage Pipeline for Breast Cancer Classification on MRI},
  author={Hamm, Benjamin and Kirchhoff, Yannick and Rokuss, Maximilian and Maier-Hein, Klaus},
  journal={arXiv preprint arXiv:2510.27326},
  year={2025}
}
```

Paper: https://arxiv.org/pdf/2510.27326

The under-the-hood breast segmentation relies on BreastDivider:

```bibtex
@article{rokuss2025breastdivider,
  title     = {Divide and Conquer: A Large-Scale Dataset and Model for Left-Right Breast MRI Segmentation},
  author    = {Rokuss, Maximilian and Hamm, Benjamin and Kirchhoff, Yannick and Maier-Hein, Klaus},
  journal   = {arXiv preprint arXiv:2507.13830},
  year      = {2025}
}
```
