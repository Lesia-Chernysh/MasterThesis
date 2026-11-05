import torch
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock as ResNetBasicBlock
from zennit.canonizers import CompositeCanonizer, SequentialMergeBatchNorm, AttributeCanonizer
from zennit.layer import Sum
from zennit.types import ConvolutionTranspose
from timm.models.resnet import Bottleneck as ResNetBottleneckTimm


class ResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        if hasattr(self, 'last_conv'):
            out = self.last_conv(out)
            out = out + 0

        out = self.relu(out)

        return out

class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if isinstance(module, ResNetBottleneckTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)
        return out

class CRPCompatibleMergeBatchNorm(SequentialMergeBatchNorm):
    """Makes eps in normalization layer > 0. Otherwise, an error arises."""
    @staticmethod
    def merge_batch_norm(modules, batch_norm):
        denominator = (
                              batch_norm.running_var + batch_norm.eps
                      ) ** 0.5

        scale = batch_norm.weight / denominator

        for module in modules:
            if module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros_like(batch_norm.bias)
                )

            index = (
                slice(None),
                *((None,) * (module.weight.ndim - 1)),
            )

            if isinstance(module, ConvolutionTranspose):
                index = index[1::-1] + index[2:]

            object.__setattr__(
                module,
                "weight",
                module.weight * scale[index],
            )

            object.__setattr__(
                module,
                "bias",
                (
                        module.bias - batch_norm.running_mean
                ) * scale + batch_norm.bias,
            )

        # Identity BatchNorm with positive eps:
        #
        # (x - 0) / sqrt(0.5 + 0.5) * 1 + 0 = x
        object.__setattr__(
            batch_norm,
            "running_mean",
            torch.zeros_like(batch_norm.running_mean),
        )
        object.__setattr__(
            batch_norm,
            "running_var",
            torch.full_like(batch_norm.running_var, 0.5),
        )
        object.__setattr__(
            batch_norm,
            "bias",
            torch.zeros_like(batch_norm.bias),
        )
        object.__setattr__(
            batch_norm,
            "weight",
            torch.ones_like(batch_norm.weight),
        )
        object.__setattr__(
            batch_norm,
            "eps",
            0.5,
        )


class PUREResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            CRPCompatibleMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
            ResNetBasicBlockCanonizer(),
        ))
