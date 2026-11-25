from zennit.composites import NameMapComposite
from zennit.core import Composite
from interpretability.attribution_methods.crp.hooks import MaskHook
from typing import Callable, List, Dict, Union, Tuple
import torch
import math
from tqdm import tqdm
from interpretability.attribution_methods.crp.attribution import CondAttribution, attrResult


class CondAttributionCNNAdapter(CondAttribution):
    def heatmap_modifier(self, data, on_device=None):
        #print("attrib, heatmap modifier")
        heatmap = data.grad.detach()
        heatmap = heatmap.to(on_device) if on_device else heatmap

        #print("heatmap size", heatmap.shape)
        #print("heatmap size after sum", torch.sum(heatmap, dim=1).shape)

        return torch.sum(heatmap, dim=1)

    def broadcast(self, data, conditions) -> Tuple[torch.Tensor, Dict]:
        #print("attrib, broadcast")
        len_data, len_cond = len(data), len(conditions)

        if len_data == len_cond:
            data.retain_grad()
            return data, conditions

        if len_cond > 1:
            data = torch.repeat_interleave(data, len_cond, dim=0)
        if len_data > 1:
            conditions = conditions * len_data

        data.retain_grad()
        return data, conditions

    def _check_arguments(self, data, conditions, start_layer, exclude_parallel, init_rel):
        return super()._check_arguments(data, conditions, start_layer, exclude_parallel, init_rel)


    def _register_mask_fn(self, hook, mask_map, b_index, c_indices, l_name):
        return super()._register_mask_fn(hook, mask_map, b_index, c_indices, l_name)


    def __call__(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = None,
            mask_map: Union[Callable, Dict[str, Callable]] = None, start_layer: str = None, init_rel=None,
            on_device: str = None, exclude_parallel=True) -> attrResult:

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

        data: torch.Tensor
            Input sample for which a conditional heatmap is computed
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

        Returns:
        --------
         
        attrResult: namedtuple object
            Contains the attributes 'heatmap', 'activations', 'relevances' and 'prediction'.
            'heatmap': torch.Tensor
                Output of the self.attribution_modifier method that defines how 'data'.grad is processed.
            'activations': dict of str and torch.Tensor
                The keys are the layer names and values are the activations
            'relevances': dict of str and torch.Tensor
                The keys are the layer names and values are the relevances
            'prediction': torch.Tensor
                The model prediction output. If 'start_layer' is set, 'prediction' is the layer activation.       
        """
        #print("attrib, __call__")

        # Use the mask belonging to the concept by default.
        if mask_map is None:
            mask_map = self.concept.mask

        record_layer = [] if record_layer is None else record_layer
        if exclude_parallel:
            return self._conditions_wrapper(data, conditions, composite, record_layer, mask_map, start_layer, init_rel, on_device, True)
        else:
            return self._attribute(data, conditions, composite, record_layer, mask_map, start_layer, init_rel, on_device, False)

    def _conditions_wrapper(self, *args):
        return super()._conditions_wrapper(*args)

    def _attribute(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = None,
            mask_map: Union[Callable, Dict[str, Callable]] = None, start_layer: str = None, init_rel=None,
            on_device: str = None, exclude_parallel=True) -> attrResult:
        """
        Computes the actual attributions as described in __call__ method docstring.
        exclude_parallel: boolean
            If set, all layer names in 'conditions' must be identical. This limitation does not apply to the __call__ method.
        """
        record_layer = [] if record_layer is None else record_layer
        #print("attribution, _attribute")
        if mask_map is None:
            mask_map = self.concept.mask

        data, conditions = self.broadcast(data, conditions)

        #print(
        #    "DATA:",
        #    "nan:", torch.isnan(data).any().item(),
        #    "inf:", torch.isinf(data).any().item(),
        #    "min:", data.min().item(),
        #    "max:", data.max().item()
        #)

        #print("conditions", conditions)

        self._check_arguments(data, conditions, start_layer, exclude_parallel, init_rel)

        hook_map, y_targets, cond_l_names = {}, [], []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)
                    if l_name not in cond_l_names:
                        cond_l_names.append(l_name)

        handles, layer_out = self._append_recording_layer_hooks(record_layer, start_layer, cond_l_names)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        data.grad = None

        # apply conditional masking to earlier received outputs for each layer
        with self._cleanup_handles(handles), mask_composite.context(self.model), composite.context(self.model) as modified:
            #print("Inside with mask_composite.context")
            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer]

                #print("START LAYER")
                #print("pred:", pred.shape,
                #      "nan:", torch.isnan(pred).any().item(),
                #      "inf:", torch.isinf(pred).any().item(),
                #      "min:", pred.min().item(),
                #      "max:", pred.max().item())

                grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)

                #print("grad_mask:", grad_mask.shape,
                #      "nan:", torch.isnan(grad_mask).any().item(),
                #      "inf:", torch.isinf(grad_mask).any().item(),
                #      "min:", grad_mask.min().item(),
                #      "max:", grad_mask.max().item())

                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)

                self.backward(
                    pred,
                    grad_mask,
                    exclude_parallel,
                    cond_l_names,
                    layer_out
                )

            else:
                pred = modified(data)

                #print(
                #    "prediction:",
                #    pred.shape,
                #    "nan:", torch.isnan(pred).any().item(),
                #    "inf:", torch.isinf(pred).any().item(),
                #    "min:", pred.min().item(),
                #    "max:", pred.max().item()
                #)

                grad_mask = self.relevance_init(
                    pred.detach().clone(),
                    y_targets,
                    init_rel
                )

                #print(
                #    "grad_mask:",
                #    grad_mask.shape,
                #    "nan:", torch.isnan(grad_mask).any().item(),
                #    "inf:", torch.isinf(grad_mask).any().item(),
                #    "min:", grad_mask.min().item(),
                #    "max:", grad_mask.max().item()
                #)

                self.backward(
                    pred,
                    grad_mask,
                    exclude_parallel,
                    cond_l_names,
                    layer_out
                )

            attribution = self.heatmap_modifier(data, on_device)

            activations, relevances = {}, {}
            if len(layer_out) > 0:
                activations, relevances = self._collect_hook_activation_relevance(layer_out, on_device)

            #print(
            #    "DATA GRAD BEFORE HEATMAP:",
            #    "nan:", torch.isnan(data.grad).any().item(),
            #    "inf:", torch.isinf(data.grad).any().item(),
            #    "min:", data.grad.min().item(),
            #    "max:", data.grad.max().item()
            #)

            #print("attribution all nan:",
            #      torch.isnan(torch.as_tensor(attribution)).all().item())
            #print("attribution, activations, relevances shape",
            #      attribution, activations, relevances)

        '''with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer]
                grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            else:
                pred = modified(data)
                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

            attribution = self.heatmap_modifier(data, on_device)

'''
        return attrResult(attribution, activations, relevances, pred)

    def generate(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = None,
            mask_map: Union[Callable, Dict[str, Callable]] = None, start_layer: str = None, init_rel=None,
            batch_size=10, on_device=None, exclude_parallel=True, verbose=True, *args) -> attrResult:
        """
        Computes several conditional attributions for single data point by broadcasting 'data' to length 'batch_size' and
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
        record_layer = [] if record_layer is None else record_layer
        if mask_map is None:
            mask_map = self.concept.mask

        self._check_arguments(data, conditions, start_layer, exclude_parallel, init_rel)

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

        data_batch = torch.repeat_interleave(data, batch_size, dim=0)
        data_batch.grad = None
        data_batch.retain_grad()
        retain_graph = True

        with self._cleanup_handles(handles), mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data_batch)
                pred = layer_out[start_layer]
                if start_layer in cond_l_names:
                    cond_l_names.remove(start_layer)

            else:
                pred = modified(data_batch)

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
                            self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)

                if b == batches-1:
                    # last batch may have len(y_targets) != batch_size. Padded part is ignored later.
                    # and backward graph is freed with retain_graph=False
                    if not start_layer:
                        y_targets.extend([y_targets[0] for i in range(batch_size-len(y_targets))])
                    batch_size = len(cond_batch)
                    retain_graph = False

                grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
                self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out, retain_graph)

                heatmap = self.heatmap_modifier(data_batch)
                activations, relevances = {}, {}
                if len(layer_out) > 0:
                    activations, relevances = self._collect_hook_activation_relevance(
                        layer_out, on_device, batch_size)

                yield attrResult(heatmap[:batch_size], activations, relevances, pred[:batch_size])

                self._reset_gradients(data_batch)
                [hook.fn_list.clear() for hook in hook_map.values()]

        if verbose:
            pbar.close()


    def _reset_gradients(self, data):
        """
        custom zero_grad() function
        """

        for p in self.model.parameters():
            p.grad = None

        data.grad = None

    def check(self):
        return None

