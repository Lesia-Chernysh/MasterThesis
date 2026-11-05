from zennit.composites import NameMapComposite
from zennit.core import Composite
from crp.hooks import MaskHook
from crp.concepts import Concept, TransformerChannelConcept
from crp.graph import ModelGraph
from typing import Callable, List, Dict, Union, Tuple
import torch
import warnings
import numpy as np
import math
from tqdm import tqdm
from collections import namedtuple
from attribution_methods.crp.attribution import CondAttribution

attrResult = namedtuple("AttributionResults", "heatmap, activations, relevances, prediction")
attrGraphResult = namedtuple("AttributionGraphResults", "nodes, connections")


class CondAttributionViLTAdapter(CondAttribution):
    def heatmap_modifier(self, inputs, on_device=None):
        # print(f"inputs: {inputs}")
        heatmap = tuple(i.grad.detach() if i.grad is not None else torch.zeros_like(i) for i in inputs)
        # print(f"heatmap: {heatmap}")
        heatmap = tuple(h.to(on_device) if on_device else h for h in heatmap)
        return heatmap

    def broadcast(self, inputs, conditions, additional_forward_kwargs) -> Tuple[torch.Tensor, Dict]:

        len_inputs, len_cond = len(inputs[0]), len(conditions)
        assert all(len(i) == len_inputs for i in inputs)

        if len_inputs == len_cond:
            for i in inputs:
                i.retain_grad()
            return inputs, conditions, additional_forward_kwargs

        if len_cond > 1:
            inputs = tuple(torch.repeat_interleave(i, len_cond, dim=0) for i in inputs)
            additional_forward_kwargs = {key: torch.repeat_interleave(val, len_cond, dim=0) for key, val in
                                         additional_forward_kwargs}
        if len_inputs > 1:
            conditions = conditions * len_inputs

        for i in inputs:
            i.retain_grad()
        return inputs, conditions, additional_forward_kwargs

    def _check_arguments(self, inputs, conditions, start_layer, exclude_parallel, init_rel):

        if not all(i.requires_grad for i in inputs):
            raise ValueError(
                "requires_grad attribute of 'inputs' must be True.")

        if self.overwrite_input_grad:
            for i in inputs:
                i.grad = None
        elif any(i.grad is not None for i in inputs):
            warnings.warn(
                "'inputs' already has a filled .grad attribute. Set to None if not intended or set 'overwrite_grad' to True.")

        distinct_cond = set()
        for cond in conditions:
            if self.MODEL_OUTPUT_NAME not in cond and start_layer is None and init_rel is None:
                raise ValueError(
                    f"Either {self.MODEL_OUTPUT_NAME} in 'conditions' or 'start_layer' or 'init_rel' must be defined.")

            if self.MODEL_OUTPUT_NAME in cond and start_layer is not None:
                warnings.warn(
                    f"You defined a condition for {self.MODEL_OUTPUT_NAME} that has no effect, since the 'start_layer' {start_layer}"
                    " is provided where the backward pass begins. If this behavior is not wished, remove 'start_layer'.")

            if exclude_parallel:

                if len(distinct_cond) == 0:
                    distinct_cond.update(cond.keys())
                elif distinct_cond ^ set(cond.keys()):
                    raise ValueError(
                        "If the 'exclude_parallel' flag is set to True, each condition dict must contain the"
                        " same layer names. (This limitation does not apply to the __call__ method)")

    def _register_mask_fn(self, hook, mask_map, b_index, c_indices, l_name, additional_forward_kwargs, rf):

        if callable(mask_map):
            mask_fn = mask_map(b_index, c_indices, l_name, additional_forward_kwargs, rf)
        elif isinstance(mask_map, Dict):
            mask_fn = mask_map[l_name](b_index, c_indices, l_name, additional_forward_kwargs, rf)
        else:
            raise ValueError("<mask_map> must be a dictionary or callable function.")

        hook.fn_list.append(mask_fn)

    def __call__(
            self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]], conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = TransformerChannelConcept.mask, start_layer: str = None,
            init_rel=None,
            on_device: str = None, exclude_parallel=True, additional_forward_kwargs: Dict[str, torch.Tensor] = {},
            rf=False) -> attrResult:

        """
        Computes conditional attributions by masking the gradient flow of PyTorch (that is replaced by zennit with relevance values).
        The relevance distribution rules (as for LRP e.g.) are described in the zennit 'composite'. Relevance can be initialized at
        the model output or 'start_layer' with the 'init_rel' argument.
        How the relevances are masked is determined by the 'conditions' as well as the 'mask_map'. In addition, 'exclude_parallel'=True,
        restricts the PyTorch gradient flow so that it does not enter into parallel layers (shortcut connections) of the layers mentioned
        in the 'conditions' dictionary.
        The name of the model output is designated with self.MODEL_OUTPUT_NAME ('y' per default) and can be used inside 'conditions'.

        Parameters:
        -----------

        inputs: torch.Tensor or tuple of torch.Tensor
            Input samples for which a conditional heatmap is computed
        conditions: list of dict
            The key of a dict are string layer names and their value is a list of integers describing the concept (channel, neuron) index.
            In general, the values are passed to the 'mask_map' function as 'concept_ids' argument.
        composite: zennit Composite
            Object that describes how relevance is distributed. Should contain a suitable zennit Canonizer.
        mask_map: dict of callable or callable
            The keys of the dict are string layer names and the values functions that implement gradient masking. If no dict is used,
            all layers are masked according to the same function.
            The 'conditions' values are passed into the function as 'concept_ids' argument.
        start_layer: (optional) str
            Layer name where to start the backward pass instead of starting at the model output.
            If set, 'init_rel' modifies the tensor at 'start_layer' instead and a condition containing self.MODEL_OUTPUT_NAME is ignored.
        init_rel: (optional) torch.Tensor, int or callable
            Initializes the relevance distribution process as described in the LRP algorithm e.g. The callable must have the signature
            callable(activations).
            Per default, relevance is initialized with the logit activation before a non-linearity.
        on_device: (optional) str
            On which device (cpu, cuda) to save the heatmap, intermediate activations and relevances.
            Per default, everything is kept on the same device as the model parameters.
        exclude_parallel: boolean
            If set, the PyTorch gradient flow is restricted so that it does not enter into parallel layers (shortcut connections)
            of the layers mentioned in the 'conditions' dictionary. Useful to get the sole contribution of a specific concept.
        additional_forward_kwargs: dict of torch.Tensor
            Additional keyword inputs to be passed to the model without computing relevance on them.

        Returns:
        --------

        attrResult: namedtuple object
            Contains the attributes 'heatmap', 'activations', 'relevances' and 'prediction'.
            'heatmap': torch.Tensor
                Output of the self.attribution_modifier method that defines how 'inputs'.grad is processed.
            'activations': dict of str and torch.Tensor
                The keys are the layer names and values are the activations
            'relevances': dict of str and torch.Tensor
                The keys are the layer names and values are the relevances
            'prediction': torch.Tensor
                The model prediction output. If 'start_layer' is set, 'prediction' is the layer activation.
        """
        if exclude_parallel:
            return self._conditions_wrapper(inputs, conditions, composite, record_layer, mask_map, start_layer,
                                            init_rel, on_device, True, additional_forward_kwargs, rf)
        else:
            return self._attribute(inputs, conditions, composite, record_layer, mask_map, start_layer, init_rel,
                                   on_device, False, additional_forward_kwargs, rf)

    def _conditions_wrapper(self, *args):
        """
        Since 'exclude_parallel'=True requires that the condition set contains only the same layer names,
        the list is divided into distinct lists that all contain the same layer name.
        """

        inputs, conditions = args[:2]

        relevances, activations = {}, {}
        heatmap, prediction = None, None

        dist_conds = self._separate_conditions(conditions)

        for dist_layer in dist_conds:

            attr = self._attribute(inputs, dist_conds[dist_layer], *args[2:])

            for l_name in attr.relevances:
                if l_name not in relevances:
                    relevances[l_name] = attr.relevances[l_name]
                    activations[l_name] = attr.activations[l_name]
                else:
                    relevances[l_name] = torch.cat([relevances[l_name], attr.relevances[l_name]], dim=0)
                    activations[l_name] = torch.cat([activations[l_name], attr.activations[l_name]], dim=0)

            if heatmap is None:
                heatmap = attr.heatmap
                prediction = attr.prediction
            else:
                heatmap = tuple(torch.cat([h1, h2], dim=0) for h1, h2 in zip(heatmap, attr.heatmap))
                prediction = torch.cat([prediction, attr.prediction], dim=0)
        # print(f"heatmap: {heatmap}\n")
        return attrResult(heatmap, activations, relevances, prediction)

    def _attribute(
            self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]], conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = TransformerChannelConcept.mask, start_layer: str = None,
            init_rel=None,
            on_device: str = None, exclude_parallel=True, additional_forward_kwargs: Dict[str, torch.Tensor] = {},
            rf=False) -> attrResult:
        """
        Computes the actual attributions as described in __call__ method docstring.
        exclude_parallel: boolean
            If set, all layer names in 'conditions' must be identical. This limitation does not apply to the __call__ method.
        """
        # print(f"record layer: {record_layer}")

        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        inputs, conditions, additional_forward_kwargs = self.broadcast(inputs, conditions, additional_forward_kwargs)

        self._check_arguments(inputs, conditions, start_layer, exclude_parallel, init_rel)

        hook_map, y_targets, cond_l_names = {}, [], []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name, additional_forward_kwargs,
                                           rf)
                    if l_name not in cond_l_names:
                        cond_l_names.append(l_name)

        handles, layer_out = self._append_recording_layer_hooks(record_layer, start_layer, cond_l_names)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()
        with mask_composite.context(self.model), composite.context(self.model) as modified:

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if start_layer:
                _ = modified(*inputs, **additional_forward_kwargs)
                pred = layer_out[start_layer]
                grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            else:
                pred = modified(*inputs, **additional_forward_kwargs)
                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            # print(f"pred: {pred}")
            attribution = self.heatmap_modifier(inputs, on_device)
            activations, relevances = {}, {}

            # print("len out", len(layer_out))
            if len(layer_out) > 0:
                activations, relevances = self._collect_hook_activation_relevance(layer_out, on_device)
            [h.remove() for h in handles]

        # print(f"act: {activations}\nrel: {relevances}\npred: {pred}\n")
        return attrResult(attribution, activations, relevances, pred)

    def generate(
            self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]], conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = TransformerChannelConcept.mask, start_layer: str = None,
            init_rel=None,
            batch_size=10, on_device=None, exclude_parallel=True, verbose=True,
            additional_forward_kwargs: Dict[str, torch.Tensor] = {}, rf=False) -> attrResult:
        """
        Computes several conditional attributions for single data point by broadcasting 'inputs' to length 'batch_size' and
        iterating through the 'conditions' list with stepsize 'batch_size'. The model forward pass is performed only once and
        the backward graph kept in memory in order to double the performance.
        Please refer to the docstring of the __call__ method.

        batch_size: int
            batch size of each forward and backward pass
        exclude_parallel: boolean
            If set, all layer names in 'conditions' must be identical. This limitation does not apply to the __call__ method.
        verbose: boolean
            If set, a progressbar is displayed.
        """

        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        self._check_arguments(inputs, conditions, start_layer, exclude_parallel, init_rel)

        # register on all layers in layer_map an empty hook
        hook_map, cond_l_names = {}, []
        for cond in conditions:
            for l_name in cond.keys():
                if l_name not in hook_map:
                    hook_map[l_name] = MaskHook([])
                if l_name != self.MODEL_OUTPUT_NAME and l_name not in cond_l_names:
                    cond_l_names.append(l_name)

        handles, layer_out = self._append_recording_layer_hooks(record_layer, start_layer, cond_l_names)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        cond_length = len(conditions)
        if cond_length > batch_size:
            batches = math.ceil(cond_length / batch_size)
        else:
            batches = 1
            batch_size = cond_length

        inputs_batched = tuple(torch.repeat_interleave(i, batch_size, dim=0) for i in inputs)
        additional_forward_kwargs = {key: torch.repeat_interleave(val, len(conditions), dim=0) for key, val in
                                     additional_forward_kwargs}
        for b in inputs_batched:
            b.grad = None
            b.retain_grad()
        retain_graph = True

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if start_layer:
                _ = modified(*inputs_batched, **additional_forward_kwargs)
                pred = layer_out[start_layer]
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)

            else:
                pred = modified(*inputs_batched, **additional_forward_kwargs)

            if verbose:
                pbar = tqdm(total=batches, dynamic_ncols=True)

            for b in range(batches):

                if verbose:
                    pbar.update(1)

                cond_batch = conditions[b * batch_size: (b + 1) * batch_size]

                y_targets = []
                for i, cond in enumerate(cond_batch):
                    for l_name, indices in cond.items():
                        if l_name == self.MODEL_OUTPUT_NAME:
                            y_targets.append(indices)
                        else:
                            self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name,
                                                   additional_forward_kwargs, rf)

                if b == batches - 1:
                    # last batch may have len(y_targets) != batch_size. Padded part is ignored later.
                    # and backward graph is freed with retain_graph=False
                    if not start_layer:
                        y_targets.extend([y_targets[0] for i in range(batch_size - len(y_targets))])
                    batch_size = len(cond_batch)
                    retain_graph = False

                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out, retain_graph)

                heatmap = self.heatmap_modifier(inputs_batched)
                activations, relevances = {}, {}
                if len(layer_out) > 0:
                    activations, relevances = self._collect_hook_activation_relevance(
                        layer_out, on_device, batch_size)

                yield attrResult(tuple(h[:batch_size] for h in heatmap), activations, relevances, pred[:batch_size])

                self._reset_gradients(inputs_batched)
                [hook.fn_list.clear() for hook in hook_map.values()]

        [h.remove() for h in handles]

        if verbose:
            pbar.close()

    def _reset_gradients(self, inputs):
        """
        custom zero_grad() function
        """

        for p in self.model.parameters():
            p.grad = None

        for i in inputs:
            i.grad = None
