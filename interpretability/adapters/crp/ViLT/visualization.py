from typing import List, Union, Dict, Tuple, Callable
import warnings
import torch
import numpy as np
import math
from collections.abc import Iterable
import concurrent.futures
import functools
import itertools
import inspect
from tqdm import tqdm
from zennit.composites import NameMapComposite, Composite
from  interpretability.attribution_methods.crp.attribution import CondAttribution
from interpretability.attribution_methods.crp.maximization import Maximization
from interpretability.attribution_methods.crp.concepts import Concept
from interpretability.attribution_methods.crp.statistics import Statistics
from interpretability.attribution_methods.crp.hooks import FeatVisHook
from interpretability.attribution_methods.crp.helper import load_maximization, load_statistics, load_stat_targets
from interpretability.attribution_methods.crp.image import vis_img_heatmap, vis_opaque_img
from interpretability.attribution_methods.crp.cache import Cache
from interpretability.adapters.crp.ViLT.hooks import FeatVisHookViLT

from interpretability.attribution_methods.crp.visualization import FeatureVisualization


class ViLTFeatureVisualization(FeatureVisualization):

    def run_distributed(self, composite: Composite, data_start, data_end, batch_size=32, checkpoint=500,
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
            hook = FeatVisHookViLT(self, concept, l_name, dict_inputs, on_device)
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
                source_indices = samples[b * batch_size: (b + 1) * batch_size]
                last_processed = int(source_indices[-1]) + 1
                inputs, multi_targets = self.get_data_concurrently(source_indices)

                target_counts = list(map(len, multi_targets))
                targets = np.asarray(list(itertools.chain.from_iterable(multi_targets)))
                if targets.size == 0:
                    continue

                for key, input_batch in inputs.items():
                    repeats = torch.as_tensor(target_counts, device=input_batch.device)
                    inputs[key] = input_batch.repeat_interleave(repeats, dim=0)
                sample_indices = np.asarray(source_indices).repeat(target_counts, axis=0)

                conditions = [{self.attribution.MODEL_OUTPUT_NAME: [target]} for target in targets]
                dict_inputs["sample_indices"] = sample_indices
                dict_inputs["targets"] = targets
                additional_forward_kwargs = {
                    "token_type_ids": inputs.token_type_ids,
                    "attention_mask": inputs.attention_mask,
                    "pixel_mask": inputs.pixel_mask,
                }
                dict_inputs["additional_forward_kwargs"] = additional_forward_kwargs

                self.attribution(
                    (inputs.pixel_values, inputs.input_embeds),
                    conditions,
                    None,
                    exclude_parallel=False,
                    additional_forward_kwargs=additional_forward_kwargs,
                )

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

    def convert_answers_to_targets(self, answers, label2id):
        targets = []

        for sample_answers in answers:
            if torch.is_tensor(sample_answers):
                if sample_answers.ndim == 0:
                    sample_answers = sample_answers.item()
                else:
                    sample_answers = sample_answers.tolist()

            if isinstance(sample_answers, (str, int, np.integer)):
                sample_answers = [sample_answers]

            sample_targets = []

            for answer in sample_answers:
                if isinstance(answer, (int, np.integer)):
                    sample_targets.append(int(answer))
                elif answer in label2id:
                    sample_targets.append(label2id[answer])

            targets.append(sample_targets)

        return targets

    def get_data_concurrently(self, indices: Union[List, np.ndarray, torch.Tensor]):

        images, questions, answers = zip(*[self.dataset[i] for i in indices])
        # print("answers:", answers)
        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True)
        targets = self.convert_answers_to_targets(
            answers,
            self.attribution.model.hf_model.config.label2id,
        )
        # targets = [[self.attribution.model.hf_model.config.label2id[label] for label in labels if label in self.attribution.model.hf_model.config.label2id] for labels in answers]

        inputs.to(self.attribution.model.hf_model.device)
        inputs["input_embeds"] = self.attribution.model.hf_model.get_input_embeddings()(
            inputs.input_ids).detach().requires_grad_(True)
        inputs.pixel_values.requires_grad_(True)

        return inputs, targets

    @torch.no_grad()
    def analyze_relevance(self, rel, layer_name, concept, data_indices, targets, additional_forward_kwargs):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """
        d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted = self.RelMax.analyze_layer(
            torch.abs(rel), concept, layer_name, data_indices, targets, additional_forward_kwargs)

        self.RelStats.analyze_layer(d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted, layer_name)

    @torch.no_grad()
    def analyze_activation(self, act, layer_name, concept, data_indices, targets, additional_forward_kwargs):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """

        # activation analysis once per sample if multi target dataset
        unique_indices = np.unique(data_indices, return_index=True)[1]

        data_indices = data_indices[unique_indices]
        act = act[unique_indices]
        targets = targets[unique_indices]

        d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted = self.ActMax.analyze_layer(
            act, concept, layer_name, data_indices, targets, additional_forward_kwargs)

        self.ActStats.analyze_layer(d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted, layer_name)

    def _load_ref_and_attribution(self, d_indices, c_id, n_indices, layer_name, attribute, rf, plot_fn, batch_size):

        # print("_load_ref_and_attribution, d_indices:", d_indices)
        inputs, _ = self.get_data_concurrently(d_indices)

        if attribute:
            heatmaps = self._attribution_on_reference(inputs, c_id, layer_name, None, rf, n_indices, batch_size)

            if callable(plot_fn):
                return plot_fn(inputs.pixel_values, heatmaps[0], rf)
            else:
                return inputs, heatmaps

        else:
            return inputs

    def _attribution_on_reference(self, inputs, concept_id: int, layer_name: str, composite, rf=False,
                                  neuron_ids: list = None, batch_size=32):

        neuron_ids = [] if neuron_ids is None else neuron_ids
        n_samples = len(inputs.input_ids)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        if rf and (len(neuron_ids) != n_samples):
            raise ValueError("length of 'neuron_ids' must be equal to the length of 'inputs'")

        img_heatmaps = []
        txt_heatmaps = []
        for b in range(batches):
            batch_slice = slice(b * batch_size, (b + 1) * batch_size)
            batch_inputs = {
                key: input_batch[batch_slice]
                for key, input_batch in inputs.items()
            }

            if rf:
                batch_neuron_ids = neuron_ids[batch_slice]
                conditions = [
                    {layer_name: {concept_id: int(neuron_id)}}
                    for neuron_id in batch_neuron_ids
                ]
                mask_map = self.layer_map[layer_name].mask_rf
            else:
                conditions = [{layer_name: [concept_id]}]
                mask_map = self.layer_map[layer_name].mask
            # initialize relevance with activation before non-linearity (could be changed in a future release)
            attr = self.attribution((batch_inputs["pixel_values"], batch_inputs["input_embeds"]), conditions, composite,
                                    mask_map=mask_map, start_layer=layer_name,
                                    on_device=self.device, exclude_parallel=False,
                                    additional_forward_kwargs={"token_type_ids": batch_inputs["token_type_ids"],
                                                               "attention_mask": batch_inputs["attention_mask"],
                                                               "pixel_mask": batch_inputs["pixel_mask"]}, rf=rf)

            img_heatmaps.extend(attr.heatmap[0].sum(1))
            txt_heatmaps.extend(attr.heatmap[1].sum(-1))

        return (img_heatmaps, txt_heatmaps)
