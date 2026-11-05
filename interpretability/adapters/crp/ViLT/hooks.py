import weakref
import functools
import torch

from zennit.core import RemovableHandle, RemovableHandleList
from attribution_methods.crp.hooks import FeatVisHook

class FeatVisHookViLT():
    '''Feature Visualization hooks for reference sampling inside forward and backward passes.'''

    def __init__(self, FV, concept, layer_name, dict_inputs, on_device):
        """
        Parameters:
            dict_inputs: contains sample_indices and targets inputs to FV.analyze_activation and FV.analyze_relevance
        """

        self.FV = FV
        self.concept = concept
        self.layer_name = layer_name
        self.dict_inputs = dict_inputs
        self.on_device = on_device

    def post_forward(self, module, input, output):
        '''Register a backward-hook to the resulting tensor right after the forward.'''

        s_indices, targets, additional_forward_kwargs = self.dict_inputs["sample_indices"], self.dict_inputs["targets"], \
        self.dict_inputs["additional_forward_kwargs"]

        if isinstance(output, tuple):
            output = output[0]

        activation = output.detach().to(self.on_device) if self.on_device else output.detach()
        self.FV.analyze_activation(activation, self.layer_name, self.concept, s_indices, targets,
                                   additional_forward_kwargs)

        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad):
            return hook_ref().backward(module, grad)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            # only if gradient required
            output[0].register_hook(wrapper)
        return output[0] if len(output) == 1 else output

    def backward(self, module, grad):
        '''Hook applied during backward-pass'''

        s_indices, targets, additional_forward_kwargs = self.dict_inputs["sample_indices"], self.dict_inputs["targets"], \
        self.dict_inputs["additional_forward_kwargs"]
        relevance = grad.detach().to(self.on_device) if self.on_device else grad.detach()
        self.FV.analyze_relevance(relevance, self.layer_name, self.concept, s_indices, targets,
                                  additional_forward_kwargs)

        return grad

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same stored_grads list.
        '''
        return self.__class__(self.FV, self.concept, self.layer_name, self.dict_inputs, self.on_device)

    def remove(self):
        pass

    def register(self, module):
        '''Register this instance by registering the neccessary forward hook to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])