from zennit.composites import EpsilonPlusFlat
from resnet50_canonizer import PUREResNetCanonizer

def get_resnet50_composite():
    return EpsilonPlusFlat(
        epsilon=1e-6,
        zero_params=["bias"],
        canonizers=[PUREResNetCanonizer()],
    )

class ReferenceEpsilonPlusFlat(EpsilonPlusFlat):
    """
    For resnet_50
    """
    def __init__(self, skip_layer, **kwargs):
        super().__init__(**kwargs)
        self.skip_layer = skip_layer

    def mapping(self, ctx, name, module):
        if name == self.skip_layer:
            return None
        return super().mapping(ctx, name, module)


