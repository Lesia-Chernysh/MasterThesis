from collections import Counter
import cv2
import matplotlib.pyplot as plt
from my_datasets.indexed_sample_dataset import IndexedSampleDataset
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# converters
def convert_to_json(obj: torch.Tensor | dict | tuple | list | int | float | str | bool):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


# select polysemantic neurons based on polysem scores
def select_polysem_neurons(scores: dict, threshold: float = 0.5):
    polysem_neurons = {}
    for layer, info in scores.items():
        for neuron_id, score in info.items():
            neuron_id = int(neuron_id)
            score = score[0]

            if score >= threshold:
                polysem_neurons[f"{layer}__{neuron_id}"] = score

    return polysem_neurons


# find polysemanticity candidates based on clusters properties
def load_crop_and_heatmap(
    record,
    cv_obj,
    layer,
    neuron_id,
    img_pos,
):
    image_path = record["image_path"]

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    ymin, ymax, xmin, xmax = (
        int(value)
        for value in record["bbox"]
    )

    crop = image[
        ymin:ymax,
        xmin:xmax,
    ]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop for record: {record}"
        )

    layer_cache = cv_obj.actmax_cache.cache[
        layer
    ]

    neuron_id = int(neuron_id)
    img_pos = int(img_pos)

    if neuron_id >= layer_cache.activations_not_aggr.shape[0]:
        raise IndexError(
            f"Neuron {neuron_id} is outside activation "
            f"shape {tuple(layer_cache.activations_not_aggr.shape)}"
        )

    if img_pos >= layer_cache.activations_not_aggr.shape[1]:
        raise IndexError(
            f"Reference position {img_pos} is outside "
            f"activation shape "
            f"{tuple(layer_cache.activations_not_aggr.shape)}"
        )

    cached_activation = (
        layer_cache.activations_not_aggr[
            neuron_id,
            img_pos,
        ]
    )

    heatmap_2d = cv_obj.adapter.create_heatmap(
        cached_activation
    )

    heatmap_2d = np.asarray(
        heatmap_2d,
        dtype=np.float32,
    )

    if heatmap_2d.ndim != 2:
        raise ValueError(
            "The adapter must return a two-dimensional "
            f"heatmap, got {heatmap_2d.shape}"
        )

    heatmap_2d = np.maximum(
        heatmap_2d,
        0,
    )

    maximum = heatmap_2d.max()

    if maximum > 0:
        heatmap_2d /= maximum

    image_height, image_width = image.shape[:2]

    full_heatmap = cv2.resize(
        heatmap_2d,
        (
            image_width,
            image_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_crop = full_heatmap[
        ymin:ymax,
        xmin:xmax,
    ]

    heatmap_uint8 = (
        np.clip(heatmap_crop, 0, 1)
        * 255
    ).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_colored = cv2.cvtColor(
        heatmap_colored,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        crop,
        0.55,
        heatmap_colored,
        0.45,
        0,
    )

    return crop, heatmap_colored, overlay


def show_neuron_clusters(
    layer_neuron,
    crop_records,
    labels,
    cv_obj,
    probabilities=None,
):
    """
    Show the original crop, activation heatmap and overlay
    for every reference, grouped by semantic cluster.
    """
    layer, neuron_id = layer_neuron.split("__")[0], layer_neuron.split("__")[1]

    labels = np.asarray(labels)

    if len(crop_records) != len(labels):
        raise ValueError(
            f"{len(crop_records)} crop records but "
            f"{len(labels)} labels"
        )

    if probabilities is not None:
        probabilities = np.asarray(
            probabilities
        )

        if len(probabilities) != len(labels):
            raise ValueError(
                f"{len(probabilities)} probabilities but "
                f"{len(labels)} labels"
            )

    unique_labels = sorted(
        np.unique(labels)
    )

    for cluster_id in unique_labels:
        indices = np.where(
            labels == cluster_id
        )[0]

        cluster_name = (
            "Noise"
            if cluster_id == -1
            else f"Cluster {cluster_id}"
        )

        print(
            f"{cluster_name}: "
            f"{len(indices)} samples"
        )

        # Each sample gets one column. The rows show:
        #   1. original crop
        #   2. heatmap
        #   3. overlay
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(indices),
            figsize=(
                3 * len(indices),
                8,
            ),
            squeeze=False,
        )

        for column, reference_index in enumerate(
            indices
        ):
            record = crop_records[
                reference_index
            ]

            (
                crop,
                heatmap,
                overlay,
            ) = load_crop_and_heatmap(
                record=record,
                cv_obj=cv_obj,
                layer=layer,
                neuron_id=neuron_id,
                img_pos=reference_index,
            )

            title = (
                f"ref={reference_index}"
            )

            if probabilities is not None:
                title += (
                    "\n"
                    f"p={probabilities[reference_index]:.2f}"
                )

            '''axes[0, column].imshow(crop)
            axes[0, column].set_title(title)

            axes[1, column].imshow(heatmap)'''
            axes[0, column].imshow(overlay)

            for row in range(1):
                axes[row, column].axis(
                    "off"
                )

        '''axes[0, 0].set_ylabel(
            "Crop",
            fontsize=11,
        )
        axes[1, 0].set_ylabel(
            "Activation",
            fontsize=11,
        )'''
        axes[0, 0].set_ylabel(
            "Overlay",
            fontsize=11,
        )

        figure.suptitle(
            f"{layer}, neuron {neuron_id}: "
            f"{cluster_name}",
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()

def find_polysem_candidates(
        cv_obj,
        polysem_neurons,
        concept_db,
        min_cluster_size,
        silhoutte_score_threshold=0.5,
        confidence_threshold=0.5,
        neurons_crops_mapping=None,
        visualize_polysem_cands: bool = False
) -> list:
    if cv_obj.crop_db is None:
        cv_obj.create_crop_db()

    if neurons_crops_mapping is None:
        neurons_crops_mapping = cv_obj.create_neuron_crops_mapping()

    polysemantic_candidates = []

    for neuron, pol_score in polysem_neurons.items():
        print(f"\nNeuron: {neuron}")
        print(f"Polysemanticity score: {pol_score}")
        layer = neuron.split("__")[0]
        neuron_id = int(neuron.split("__")[1])

        embeddings = concept_db[layer][neuron_id]
        embeddings = normalize(
            embeddings.detach().cpu().numpy()
        )

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            copy=False
        )

        labels = clusterer.fit_predict(embeddings)


        # Check number of clusters and noise samples
        print(cv_obj.actmax_cache.cache[layer].activations[neuron_id])
        print("Labels:", labels)
        print("Cluster sizes:", Counter(labels))

        cluster_ids = sorted(
            cluster_id
            for cluster_id in np.unique(labels)
            if cluster_id != -1
        )

        print("Number of clusters:", len(cluster_ids))
        print("Number of noise samples:", (labels == -1).sum())

        probabilities = clusterer.probabilities_
        print(probabilities)

        cluster_prob = dict.fromkeys(cluster_ids, 0)
        for index, (
            label,
            probability,
        ) in enumerate(zip(labels, probabilities)):
            if label != -1:
                cluster_prob[label] += probability

        for cluster, prob in cluster_prob.items():
            cluster_prob[cluster] = prob / np.count_nonzero(labels == int(cluster))

        # Check confidence
        count_confident_classes = 0

        for cluster, mean_prob in cluster_prob.items():
            if mean_prob >= confidence_threshold:
                count_confident_classes += 1

        # Calculate silhouette score
        non_noise_mask = labels != -1
        non_noise_labels = labels[
            non_noise_mask
        ]

        if (
            len(np.unique(non_noise_labels)) >= 2
            and non_noise_mask.sum()
            > len(np.unique(non_noise_labels))
        ):
            score = silhouette_score(
                embeddings[non_noise_mask],
                non_noise_labels,
                metric="euclidean",
            )
        else:
            score = None

        print(f"silhouette score: {score}")

        # Summerize, if the neuron is a polysemantic candidate
        is_polysemantic_candidate = (
            len(cluster_ids) >= 2
            and count_confident_classes >= 2
            and (score is not None and score >= silhoutte_score_threshold)
        )

        if is_polysemantic_candidate:
            crop_ids = neurons_crops_mapping[
                layer
            ][neuron_id]
            assert len(crop_ids) == len(embeddings), (
                f"{len(crop_ids)} crop IDs but "
                f"{len(embeddings)} embeddings"
            )

            if visualize_polysem_cands:
                crop_records = [
                    cv_obj.crop_db[crop_id]
                    for crop_id in crop_ids
                ]

                show_neuron_clusters(
                   neuron,
                    crop_records,
                    labels,
                    cv_obj,
                   clusterer.probabilities_,
                )

            # store candidate results
            candidate_result = {
                "layer": layer,
                "neuron_id": int(neuron_id),
                "n_references": len(embeddings),
                "n_clusters": len(cluster_ids),
                "n_noise": int((labels == -1).sum()),
                "labels": labels.tolist(),
                "probabilities": (
                    clusterer.probabilities_.tolist()
                ),
                "silhouette_score": (
                    None
                    if score is None
                    else float(score)
                ),
                "crop_ids": [
                    int(crop_id)
                    for crop_id in crop_ids
                ],
            }
            polysemantic_candidates.append(candidate_result)


    return polysemantic_candidates


# collect activation patterns of layers based on selected samples
def collect_activation_patterns(
    model,
    adapter,
    dataset,
    sample_ids,
    layer_names,
    device,
    batch_size=32,
    num_workers=0,
    reduction="mean",
):
    """
    Collect full-layer activation patterns.

    Parameters
    ----------
    reduction : {"mean", "max", "cls", "tokens"}
        mean:
            Average over all spatial/token dimensions.
            Output per layer: [samples, neurons].

        max:
            Maximum over all spatial/token dimensions.
            Output per layer: [samples, neurons].

        cls:
            Use the CLS-token activation.
            Output per layer: [samples, neurons].
            Intended for ViT only.

        tokens:
            Retain every token activation.
            Output per layer:
                [samples, neurons, tokens].

            This requires substantially more memory.

    Returns
    -------
    dict
        {
            layer_name: {
                "sample_ids": Tensor[num_samples],
                "activations": Tensor[...]
            }
        }
    """
    if reduction not in {
        "mean",
        "max",
        "cls",
        "tokens",
    }:
        raise ValueError(
            f"Unknown reduction: {reduction}"
        )

    indexed_dataset = IndexedSampleDataset(
        dataset=dataset,
        sample_ids=sample_ids,
    )

    dataloader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    named_modules = dict(
        model.named_modules()
    )

    missing_layers = (
        set(layer_names)
        - set(named_modules)
    )

    if missing_layers:
        raise ValueError(
            "The following layers do not exist in "
            f"model.named_modules(): {missing_layers}"
        )

    collected = {
        layer_name: {
            "sample_ids": [],
            "activations": [],
        }
        for layer_name in layer_names
    }

    # Hooks need access to information about the batch that
    # is currently passing through the model.
    current_batch = {
        "sample_ids": None,
        "prepared_batch": None,
    }

    handles = []

    def create_hook(layer_name):
        def hook_fn(
            module,
            inputs,
            output,
        ):
            prepared_batch = current_batch[
                "prepared_batch"
            ]
            batch_sample_ids = current_batch[
                "sample_ids"
            ]

            if prepared_batch is None:
                raise RuntimeError(
                    "Hook executed before the adapter "
                    "prepared the current batch"
                )

            # Convert model-specific hook output to:
            #
            #   [batch, neurons, *spatial]
            #
            # For ViT this should normally be:
            #
            #   [batch, neurons, tokens]
            canonical_activations = (
                adapter.canonicalize_activations(
                    output,
                    prepared_batch,
                )
            )

            if (
                canonical_activations.shape[0]
                != len(batch_sample_ids)
            ):
                raise ValueError(
                    f"Layer {layer_name!r} returned batch "
                    f"size "
                    f"{canonical_activations.shape[0]}, "
                    f"but there are "
                    f"{len(batch_sample_ids)} sample IDs"
                )

            if reduction == "tokens":
                activation_patterns = (
                    canonical_activations
                )

            elif reduction == "mean":
                activation_patterns = (
                    canonical_activations
                    .flatten(start_dim=2)
                    .mean(dim=2)
                )

            elif reduction == "max":
                activation_patterns = (
                    canonical_activations
                    .flatten(start_dim=2)
                    .amax(dim=2)
                )

            elif reduction == "cls":
                if (
                    canonical_activations.ndim
                    != 3
                ):
                    raise ValueError(
                        "CLS reduction requires ViT-like "
                        "activations with shape "
                        "[batch, neurons, tokens]"
                    )

                # Canonical ViT representation:
                # [batch, neurons, tokens]
                activation_patterns = (
                    canonical_activations[
                        :,
                        :,
                        0,
                    ]
                )

            collected[layer_name][
                "sample_ids"
            ].append(
                batch_sample_ids.detach().cpu()
            )

            collected[layer_name][
                "activations"
            ].append(
                activation_patterns
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )

        return hook_fn

    for layer_name in layer_names:
        handle = named_modules[
            layer_name
        ].register_forward_hook(
            create_hook(layer_name)
        )

        handles.append(handle)

    model.eval()
    model.to(device)

    try:
        with torch.inference_mode():
            for batch in tqdm(
                dataloader,
                desc="Collecting activation patterns",
            ):
                batch_sample_ids = batch[0]
                images = batch[1]

                prepared_batch = (
                    adapter.prepare_batch(
                        images,
                        device,
                    )
                )

                current_batch[
                    "sample_ids"
                ] = batch_sample_ids

                current_batch[
                    "prepared_batch"
                ] = prepared_batch

                # Exactly one forward pass.
                adapter.forward(
                    model,
                    prepared_batch,
                )

    finally:
        # Hooks must be removed even if inference fails.
        for handle in handles:
            handle.remove()

        current_batch["sample_ids"] = None
        current_batch[
            "prepared_batch"
        ] = None

    # Concatenate batches.
    results = {}

    for layer_name in layer_names:
        if not collected[layer_name][
            "activations"
        ]:
            raise RuntimeError(
                f"Hook for {layer_name!r} did not run"
            )

        layer_sample_ids = torch.cat(
            collected[layer_name][
                "sample_ids"
            ],
            dim=0,
        )

        layer_activations = torch.cat(
            collected[layer_name][
                "activations"
            ],
            dim=0,
        )

        if len(layer_sample_ids) != len(
            sample_ids
        ):
            raise RuntimeError(
                f"Layer {layer_name!r} collected "
                f"{len(layer_sample_ids)} samples, "
                f"expected {len(sample_ids)}"
            )

        results[layer_name] = {
            "sample_ids": layer_sample_ids,
            "activations": layer_activations,
            "sample_id_to_position": {
                int(sample_id): position
                for position, sample_id
                in enumerate(
                    layer_sample_ids.tolist()
                )
            },
        }

        print(
            f"{layer_name}: activations "
            f"{tuple(layer_activations.shape)}"
        )

    return results


# Select negative samples for clusters
def select_negatives(
    clip_embeddings,
    top50_indices,
    cluster_labels,
    target_cluster,
    seed=42
):
    """
    Select hard negatives from other HDBSCAN clusters
    of the same neuron's top-50 activating images.

    CLIP embeddings must correspond to top50_indices
    in exactly the same order.
    """

    rng = np.random.default_rng(seed)

    # Convert torch tensor to NumPy
    if isinstance(clip_embeddings, torch.Tensor):
        E = clip_embeddings.detach().cpu().numpy()
    else:
        E = np.asarray(clip_embeddings)

    E = normalize(E, axis=1)

    top50_indices = np.asarray(top50_indices).ravel()
    labels = np.asarray(cluster_labels).ravel()
    target_cluster = np.asarray(target_cluster).item()

    if not (
        len(E) == len(top50_indices) == len(labels)
    ):
        raise ValueError(
            "CLIP embeddings, indices and labels "
            "must have the same number of samples."
        )

    if target_cluster == -1:
        raise ValueError("Noise is not a valid target cluster.")

    # Local indices (0 to 49)
    positive_local = np.flatnonzero(
        labels == target_cluster
    )

    negative_local = np.flatnonzero(
        (labels != target_cluster) &
        (labels != -1)
    )

    if len(positive_local) == 0:
        raise ValueError("No positive samples.")

    if len(negative_local) == 0:
        raise ValueError("No other clusters available.")

    # Randomize tie-breaking
    positive_local = rng.permutation(positive_local)
    negative_local = rng.permutation(negative_local)

    # CLIP cosine similarity matrix
    similarity = (
        E[positive_local] @ E[negative_local].T
    )

    # Find optimal one-to-one matching
    # Maximizing similarity = minimizing -similarity
    rows, cols = linear_sum_assignment(-similarity)

    matched_positive_local = positive_local[rows]
    matched_negative_local = negative_local[cols]

    return {
        "positive_local": matched_positive_local,
        "negative_local": matched_negative_local,
        "positive_global": top50_indices[
            matched_positive_local
        ],
        "negative_global": top50_indices[
            matched_negative_local
        ],
        "similarities": similarity[rows, cols]
    }


def estimate_feature_direction(
    positive_activations,
    control_activations,
):
    direction = (
        positive_activations.mean(dim=0)
        - control_activations.mean(dim=0)
    )

    return F.normalize(
        direction,
        dim=0,
    )

