from typing import List, Union, Dict, Tuple, Callable
import warnings
import torch
import numpy as np
import math
from collections.abc import Iterable
import concurrent.futures
import functools
import inspect
from tqdm import tqdm
from zennit.composites import NameMapComposite, Composite
from interpretability.adapters.crp.CNNs_ViT.attribution import CondAttribution
from  interpretability.attribution_methods.crp.maximization import Maximization
from  interpretability.attribution_methods.crp.concepts import ChannelConcept, TransformerChannelConcept, Concept
from  interpretability.attribution_methods.crp.statistics import Statistics
from  interpretability.attribution_methods.crp.hooks import FeatVisHook
from  interpretability.attribution_methods.crp.helper import load_maximization, load_statistics, load_stat_targets
from  interpretability.attribution_methods.crp.image import vis_img_heatmap, vis_opaque_img
from  interpretability.attribution_methods.crp.cache import Cache
from  interpretability.attribution_methods.crp.visualization import FeatureVisualization

'''
        fv = PredictionFeatureVisualization(concept, attribution, imagenet_data, layer_map,
                                            preprocess_fn=None, # do we need it?
                                            path=fv_path, device=device,
                                            predicted_targets=predicted_targets # don't need
                                            )
fv = ViLTFeatureVisualization(
            attribution,
            imagenet_data_vilt,
            layer_map,
            processor,  # make optional
            path=fv_path,
            device=device)
        # TODO: add preprocess_fn=None`?
'''

class FeatureVisualizationCNNsAdapter(FeatureVisualization):
    # TODO is absend from ViLT & Qwen
    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.Tensor, int]:
        """
        returns a data sample from dataset at index.

        Parameter:
            index: integer
            preprocessing: boolean.
                If True, return the sample after preprocessing. If False, return the sample for plotting.
        """

        data, target = self.dataset[index]
        data = data.to(self.device).unsqueeze(0)
        if preprocessing:
            data = self.preprocess_data(data)

        data.requires_grad = True
        return data, target

    def run_distributed(self, composite: Composite, data_start, data_end, batch_size=16, checkpoint=500,
                        on_device=None):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        if data_end <= data_start:
            raise ValueError("data_end must be greater than data_start.")
        if batch_size <= 0 or checkpoint <= 0:
            raise ValueError("batch_size and checkpoint must be positive integers.")

        self.saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}
        last_checkpoint = data_start

        n_samples = data_end - data_start
        samples = np.arange(start=data_start, stop=data_end)

        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        # feature visualization is performed inside forward and backward hook of layers
        name_map, dict_inputs = [], {}
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        pbar = tqdm(total=batches, dynamic_ncols=True)
        composite_registered = False
        fv_registered = False
        last_processed = data_start
        try:
            if composite:
                composite.register(self.attribution.model)
                composite_registered = True
            fv_composite.register(self.attribution.model)
            fv_registered = True

            for b in range(batches):
                pbar.update(1)
                samples_batch = samples[b * batch_size: (b + 1) * batch_size]
                last_processed = int(samples_batch[-1]) + 1
                data_batch, targets_samples = self.get_data_concurrently(samples_batch, preprocessing=True)
                targets_samples = np.atleast_1d(np.asarray(targets_samples))

                data_broadcast, targets, sample_indices = [], [], []
                try:
                    for i_t, target in enumerate(targets_samples):
                        for single_target in self.multitarget_to_single(target):
                            targets.append(single_target)
                            data_broadcast.append(data_batch[i_t])
                            sample_indices.append(samples_batch[i_t])
                    if not data_broadcast:
                        continue
                    data_broadcast = torch.stack(data_broadcast, dim=0)
                    sample_indices = np.asarray(sample_indices)
                    targets = np.asarray(targets)
                except NotImplementedError:
                    data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

                conditions = [{self.attribution.MODEL_OUTPUT_NAME: [target]} for target in targets]
                dict_inputs["sample_indices"] = sample_indices
                dict_inputs["targets"] = targets
                self.attribution(data_broadcast, conditions, None, exclude_parallel=False)

                if b % checkpoint == checkpoint - 1:
                    self._save_results((last_checkpoint, last_processed))
                    last_checkpoint = last_processed

            if last_processed > last_checkpoint:
                self._save_results((last_checkpoint, last_processed))
        finally:
            if fv_registered:
                fv_composite.remove()
            if composite_registered:
                composite.remove()
            pbar.close()

        return self.saved_checkpoints

    @torch.no_grad()
    def analyze_relevance(self, rel, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """
        # print("analyse relevance, FV")
        # TODO: vilt has additional forw. keys
        d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted = self.RelMax.analyze_layer(
            rel, concept, layer_name, data_indices, targets)

        # print("t_c_sorted shape", t_c_sorted.shape)

        self.RelStats.analyze_layer(d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted, layer_name)

    @torch.no_grad()
    def analyze_activation(self, act, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """

        # activation analysis once per sample if multi target dataset
        unique_indices = np.unique(data_indices, return_index=True)[1]
        data_indices = data_indices[unique_indices]
        act = act[unique_indices]
        targets = targets[unique_indices]

        d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted = self.ActMax.analyze_layer(
            act, concept, layer_name, data_indices, targets)

        self.ActStats.analyze_layer(d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted, layer_name)

    def get_data_concurrently(self, indices: Union[List, np.ndarray, torch.Tensor], preprocessing=False):

        if len(indices) == 1:
            data, label = self.get_data_sample(indices[0], preprocessing)
            return data, [label]

        threads = []
        data_returned = []
        labels_returned = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for index in indices:
                future = executor.submit(self.get_data_sample, index, preprocessing)
                threads.append(future)

        for t in threads:
            single_data = t.result()[0]
            single_label = t.result()[1]
            data_returned.append(single_data)
            labels_returned.append(single_label)

        data_returned = torch.cat(data_returned, dim=0)
        return data_returned, labels_returned


    def _load_ref_and_attribution(self, d_indices, c_id, n_indices, layer_name, composite, rf, plot_fn, batch_size):

        data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=False)

        if composite:
            # print("composite", composite)
            data_p = self.preprocess_data(data_batch)
            heatmaps = self._attribution_on_reference(data_p, c_id, layer_name, composite, rf, n_indices, batch_size)

            if callable(plot_fn):
                return plot_fn(data_batch.detach(), heatmaps.detach(), rf)
            else:
                return data_batch.detach().cpu(), heatmaps.detach().cpu()

        else:
            return data_batch.detach().cpu()



    def _attribution_on_reference(self, data, concept_id: int, layer_name: str, composite, rf=False,
                                  neuron_ids: list = None, batch_size=32):
        neuron_ids = [] if neuron_ids is None else neuron_ids
        # print("_attribution_on_reference")
        # print("concept_id", concept_id)

        n_samples = len(data)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        if rf and (len(neuron_ids) != n_samples):
            raise ValueError("length of 'neuron_ids' must be equal to the length of 'data'")

        heatmaps = []
        for b in range(batches):
            data_batch = data[b * batch_size: (b + 1) * batch_size].detach().requires_grad_()

            if rf:
                batch_neuron_ids = neuron_ids[
                    b * batch_size: (b + 1) * batch_size
                ]

                conditions = [
                    {layer_name: {concept_id: n_index}}
                    for n_index in batch_neuron_ids
                ]
                mask_map = self.layer_map[layer_name].mask_rf
            else:
                conditions = [{layer_name: [concept_id]}]
                mask_map = self.layer_map[layer_name].mask

            attr = self.attribution(
                data_batch,
                conditions,
                composite,
                mask_map=mask_map,
                start_layer=layer_name,
                on_device=self.device,
                exclude_parallel=False,
            )

            heatmaps.append(attr.heatmap)

        return torch.cat(heatmaps, dim=0)
