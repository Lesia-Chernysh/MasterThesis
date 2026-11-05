'''
Algorithm:
1. create a model, device, fm_model

2. define concept  --> cc / tcc

3.* define composite --> resnet, vit, vilt, qwen

4. select layers, create layer map

5. define transform, preprocessing for the dataset

6. download dataset --> for ViLT, Qwen VQA

7. init attribution -->

8. init feature visualization  --> possibility to run on a subset

9. run crp


------------ crp: ----------------
different in visualization
* run distributed
* get_data_concurrently
* analyze_relevance

hooks
'''
import torch
from attribution_methods.crp.attribution import CondAttribution
from attribution_methods.crp.concepts import ChannelConcept
from adapters.crp.CNNs_ViT.attribution import CondAttributionCNNAdapter
from models.create_model import create_resnet50

resnet50 = create_resnet50()

CNNfv = CondAttributionCNNAdapter(
    concept=ChannelConcept(),
    model=resnet50,
    device=torch.device('cpu'),
)

CNNfv.check()
