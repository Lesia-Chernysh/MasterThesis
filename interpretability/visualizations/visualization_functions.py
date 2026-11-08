import json
import math
from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt
from my_datasets.imagenet_mini import ImagenetMini, ImagenetMiniVQA, ImagenetMiniPredLabels
import torch
import csv
import cv2
import numpy as np

from configs.general_cofig import *
# --------------------------------------------------- Config -------------------------------------------------------
THRESHOLD_ACT = 0.5
# ---------------------------------------------------------------------------------------------------------------------


# --------------- For dataset samples -----------------
def visualize_dataset_sample(sample: tuple[Image.Image, str|torch.Tensor]) -> None:
    '''
    Visualizes images from a dataset.
    :param samples: list[tuple(image, label)]
    :return: None
    '''

    # visualize samples from the dataset
    img = sample[0]
    label = sample[-1]

    plt.imshow(img)

    if len(sample) == 3:
        question = sample[1]
        plt.title(f"Class: {label}. Question: {question}")
    else:
        plt.title(f"Class: {label}.")

    plt.axis('off')
    plt.show()


# ----------------------------- Statistics -------------------------
# Activation score distribution
def report_and_plot_cached_activation_scores(
    activation_cache_path,
    output_dir,
    bins=40,
):
    """Report and plot per-layer neuron scores from the ActMax cache.

    A neuron's score is the mean of its retained top-reference activation
    values, which is also the score used by ``global_top_neurons``. Therefore
    these are distributions over neurons, not over every image activation in
    the dataset.

    Parameters
    ----------
    activation_cache_path:
        Path to an npz: ``all_visual_mlp_actmax.npz`` or pt file: "top_500_per_layer_*model*.pt".
    output_dir:
        Directory in which CSV and PNG outputs are written.
    bins:
        Number of histogram bins.

    Returns
    -------
    dict
        ``{layer_name: {mean, min, max, neuron_count}}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if activation_cache_path.split(".")[-1] == "npz":
        print("Input is a .npz file")
        activation_cache = np.load(activation_cache_path)

        layer_names = activation_cache["layer_names"].astype(str)
        reference_count = int(activation_cache["reference_count"])
        top_values = activation_cache["top_values"][:, :reference_count, :]

        neuron_scores = np.mean(top_values, axis=1)  # layers x neurons

    elif activation_cache_path.split(".")[-1] == "pt":
        print("Input is a .pt file")
        activation_cache = torch.load(activation_cache_path)

        layer_names = list(activation_cache.keys())
        reference_count = 10
        neuron_scores = []
        for l, stats in activation_cache.items():
            per_layer = []
            for _, acts in stats.items():
                per_layer.append(acts)
            neuron_scores.append(np.array(per_layer))


    statistics = {}
    print(f"\n{'=' * 72}\nCached activation-score statistics\n{'=' * 72}")
    print(
        "Each value is one neuron's mean activation over its "
        f"top {reference_count} cached references."
    )
    for layer_index, layer in enumerate(layer_names):
        values = neuron_scores[layer_index]
        values = values[np.isfinite(values)]
        neuron_count = int(values.size)
        layer_stats = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "neuron_count": neuron_count,
        }
        statistics[layer] = layer_stats
        print(
            f"{layer}: n={layer_stats['neuron_count']}, "
            f"mean={layer_stats['mean']:.6g}, "
            f"min={layer_stats['min']:.6g}, "
            f"max={layer_stats['max']:.6g}"
        )

    csv_path = output_dir / "activation_score_statistics_per_layer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "neuron_count", "mean", "min", "max"])
        for layer, values in statistics.items():
            writer.writerow([
                layer, values["neuron_count"], values["mean"],
                values["min"], values["max"],
            ])

    panels = len(layer_names) + 1
    columns = min(4, panels)
    rows = math.ceil(panels / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False
    )
    axes = axes.reshape(-1)
    for layer_index, (axis, layer) in enumerate(zip(axes, layer_names)):
        values = neuron_scores[layer_index]
        values = values[np.isfinite(values)]
        axis.hist(values, bins=bins)
        axis.axvline(values.mean(), color="red", linestyle="--", linewidth=1,
                     label=f"mean={values.mean():.3g}")
        axis.set(title=layer, xlabel="Mean top-reference activation", ylabel="Neurons")
        axis.legend(fontsize=8)
    if activation_cache_path.split(".")[-1] == "npz":
        all_values = neuron_scores[np.isfinite(neuron_scores)]
    else:
        all_values = np.concatenate(neuron_scores)
        all_values = all_values[np.isfinite(all_values)]
    overall = axes[len(layer_names)]
    overall.hist(all_values, bins=bins)
    overall.axvline(all_values.mean(), color="red", linestyle="--", linewidth=1,
                    label=f"mean={all_values.mean():.3g}")
    overall.set(title="All visual MLP layers", xlabel="Mean top-reference activation",
                ylabel="Neurons")
    overall.legend(fontsize=8)
    for axis in axes[panels:]:
        axis.axis("off")
    figure.suptitle("Cached neuron activation-score distributions")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path = output_dir / "activation_score_distributions_per_layer.png"
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    print(f"Saved activation statistics to {csv_path}")
    print(f"Saved activation distributions to {plot_path}")
    return statistics

# Polysemanticity score distribution
def report_polysem(
        polysem_scores,
        title=f"Polysemanticity score: global top {K_ACT_NEURONS} neurons",
        k_polysem_n=5
):
    all_scores = [score for neurons in polysem_scores.values()
                  for score in neurons.values()]
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(f"neurons: {len(all_scores)}")
    try:
        print(f"general polysemanticity level: {float(np.mean(all_scores.cpu())):.6f}")
        print(f"median polysemanticity: {float(np.median(all_scores.cpu())):.6f}")
    except AttributeError:
        print(f"general polysemanticity level: {float(np.mean(all_scores)):.6f}")
        print(f"median polysemanticity: {float(np.median(all_scores)):.6f}")
    for layer, neurons in polysem_scores.items():
        ordered = sorted(neurons.items(), key=lambda item: item[1], reverse=True)
        print(f"\n{layer}: n={len(ordered)}, mean={np.mean(list(neurons.values())):.6f}")
        print("  most polysemantic:")
        for neuron, score in ordered[:k_polysem_n]:
            if torch.is_tensor(score):
                score = float(score.detach().cpu().item())
            print(f"    {neuron}: {score:.6f}")
        print("  least polysemantic:")
        for neuron, score in reversed(ordered[-k_polysem_n:]):
            if torch.is_tensor(score):
                score = float(score.detach().cpu().item())
            print(f"    {neuron}: {score:.6f}")


def plot_polysem(
        polysem_scores,
        title=f"Polysemanticity score: global top {K_ACT_NEURONS} neurons",
        output_path=None,
        bins=20
):
    def to_numpy(value):
      if isinstance(value, torch.Tensor):
          return value.detach().cpu().numpy()
      return np.asarray(value, dtype=float)
      
    layers = list(polysem_scores)
    panels = len(layers) + 1
    columns = min(4, panels)
    rows = math.ceil(panels / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
    )
    axes = axes.ravel()

    all_scores = []

    for axis, layer in zip(axes, layers):
        # Flatten nested lists/arrays into one numeric sequence.
        values = np.concatenate([
            to_numpy(v).ravel()
            for v in polysem_scores[layer].values()
        ])
        # Optional: discard NaN and infinite values.
        values = values[np.isfinite(values)]
        all_scores.extend(values.tolist())

        axis.hist(values, bins=bins, color="tab:blue")
        axis.axvline(
            0.5,
            color="red",
            linestyle="--",
            linewidth=1,
        )
        axis.set(
            title=str(layer),
            xlabel="Score",
            ylabel="Count",
        )

    overall = axes[len(layers)]
    overall.hist(all_scores, bins=bins, color="tab:blue")
    overall.axvline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=1,
        label="CRP threshold",
    )
    overall.set(
        title="All represented layers",
        xlabel="Score",
        ylabel="Count",
    )
    overall.legend()

    for axis in axes[panels:]:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.show()
    plt.close(figure)

    print(f"All scores: {len(all_scores)}")
    if output_path is not None:
        print(f"Saved distribution plot to {output_path}")
        print(f"Saved distribution plot to {output_path}")


# -------------------- ActMax reference samples -------------------------
def restore_crop(img_path, bbox):
    """
    Creates an array of a crop based on the original image and bounding box.
    :param img_path:
    :param bbox:
    :return:
    """
    ymin, ymax, xmin, xmax = bbox
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    return orig_img[ymin:ymax, xmin:xmax]

def _parse_neuron_id(n_id: str | int, tgt_dict: dict) -> int | str:
    target_type = type(list(tgt_dict.keys())[0])

    if type(n_id) != target_type:
        if target_type == str:
            return str(n_id)
        elif target_type == int:
            return int(n_id)
        else:
            print(f"Unsupported target type {target_type} for neuron id in polysem scores")
    return n_id


def _visualize_activations(cv_object, dataset, images_path: list, sample_ids, layer, neuron_id):
    def _parse_image(img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)

        elif torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = np.asarray(img)

        # C × H × W -> H × W × C
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.moveaxis(img, 0, -1)

        return img

    n_images = len(images_path)

    fig, axes = plt.subplots(
        4, n_images,
        figsize=(3 * n_images, 12),
        squeeze=False
    )

    row_titles = [
        "Original Image",
        "Activation Map (Overlay)",
        "Masked Output",
        "Cropped Output"
    ]

    for i in range(n_images):
        img_path = images_path[i][0]
        sample_id = sample_ids[i]

        #orig_img = cv2.imread(img_path)
        orig_img = dataset[sample_id][0]
        orig_img = _parse_image(orig_img)
        #orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = orig_img.shape[:2]

        heatmap = cv2.imread(
            f"{OUTPUTS}/act_heatmaps/heatmaps_{cv_object.model_name}/{layer}_n{neuron_id}_s{sample_id}.jpeg",
            cv2.IMREAD_GRAYSCALE
        )

        heatmap_array = heatmap.astype(np.float32) / 255.0

        neuron_id = _parse_neuron_id(neuron_id, cv_object.crop_info[layer])
        bbox = cv_object.crop_info[layer][neuron_id][i][1]
        cropped_img = restore_crop(img_path, bbox)

        # Get mask
        binary_patch_mask = (
            heatmap_array >= THRESHOLD_ACT
        ).astype(np.uint8)

        # Safety: make sure at least one patch exists
        if binary_patch_mask.sum() == 0:
            top_k = 5

            flat_idx = np.argsort(
                heatmap_array.flatten()
            )[-top_k:]

            binary_patch_mask = np.zeros_like(
                heatmap_array, dtype=np.uint8
            )

            for idx in flat_idx:
                y, x = np.unravel_index(
                    idx, heatmap_array.shape
                )
                binary_patch_mask[y, x] = 1

        # Connected components
        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary_patch_mask,
                connectivity=8
            )
        )

        if num_labels > 1:
            largest_label = (
                1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            )

            component_mask = (
                labels == largest_label
            ).astype(np.uint8)
        else:
            component_mask = binary_patch_mask

        # Full-resolution mask
        mask_img = cv2.resize(
            component_mask,
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )

        mask_img_3d = mask_img[..., None]

        masked_img = orig_img * mask_img_3d

        # Resize heatmap for overlay
        heatmap_resized = cv2.resize(
            heatmap,
            (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR
        )

        # Class name for column title
        class_name = dataset[sample_id][1]

        axes[0, i].set_title(
            f"{class_name}\nSample {sample_id}",
            fontsize=10
        )

        # Row 1: Original
        axes[0, i].imshow(orig_img)

        # Row 2: Overlay
        axes[1, i].imshow(orig_img)
        axes[1, i].imshow(
            heatmap_resized,
            cmap="jet",
            alpha=0.5
        )

        # Row 3: Mask
        axes[2, i].imshow(masked_img)

        # Row 4: Crop
        axes[3, i].imshow(cropped_img)

        for row in range(4):
            axes[row, i].axis("off")

    # Add row labels on the left
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(
            title,
            fontsize=12,
            rotation=90,
            labelpad=15
        )
        axes[row, 0].yaxis.set_label_coords(-0.1, 0.5)

    fig.suptitle(
        f"Layer: {layer} | Neuron: {neuron_id}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig


def visualize_neuron(cv_object,
                     dataset,
                     polysem_scores_model: dict,
                     #top_neurons_act: dict,
                     layer: str,
                     neuron_id: int,
                     print_info: bool
                     ):
    """
    Visualizes one neuron by plotting reference samples.
    :param cv_object:
    :param polysem_scores_model:
    :param layer:
    :param neuron_id:
    :param print_info:
    :return:
    """
    sample_ids = cv_object.actmax_cache.cache[layer].sample_ids

    sample_ids_to_visualize = sample_ids[neuron_id].tolist()
    images_path = [dataset.dataset.samples[i] for i in sample_ids_to_visualize]

    if print_info:
        acts_for_refs = cv_object.actmax_cache.cache[layer].activations[neuron_id]
        neuron_id = _parse_neuron_id(neuron_id, polysem_scores_model[layer])
    
        pol_score = polysem_scores_model[layer][neuron_id]
        
        if type(pol_score) is list:
            pol_score = pol_score[0]
        print(f"--- {layer}, {neuron_id} neuron ---")
        print(f"Mean activation: {float(acts_for_refs.mean())}")
        print(f"Polysemanticity score: {round(float(pol_score), 4)}")

    _visualize_activations(cv_object, dataset, images_path, sample_ids_to_visualize, layer, neuron_id)


# ------------------------ RelMax reference samples ----------------------------

# --------------------- PURE reference samples -----------------------------------


# ---------------------- All together --------------------------------------------
