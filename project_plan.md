frameworks /

    semantic_lens/
        output_files/
            ...
    crp/
        common/
            interfaces.py
            results.py
        implementations/
            original/
                attribution.py
                maximization.py
            attention_lrp/
                attribution.py
                maximization.py
            multimodal/
                attribution.py
                maximization.py
    pure/
  
**very general**

* load datasets
* load models


**semantic_lens**

**---run**

*What is the same*
* run forward pass
* register hooks
* cache activations

*What is different*
* image preprocessing
* feature visualization method:
  * CNNs: FeatureVisualization with cc
  * ViT: FeatureVisualization with tcc
  * ViLT: tcc + different input size
  * Qwen: tcc + even more different input size


***What is used in other methods***
=> always same input, output
* embedding of reference images
* compute polysem score
* plot, ananlyse polysem

* ideally, should be able to handle both cache options: 
  * from SemanticLens
  * from CRP


** visualizations **

* selected neurons (top n)
  * distribution across the layers
* neuron
  * activation-based
  * rel-based
  * pure
  * all together
* activations
  * distribution
* polysem scores
  * statistics
  * plots
* cluster references

* helpers
  * get labels for images
  * get labels for clusters ?