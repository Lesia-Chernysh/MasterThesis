from functools import partial
from torch.nn import Dropout, GELU
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLMLP, Qwen2MLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm # in transformers v5 this will be Qwen2_5_VLRMSNorm

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward, non_linear_forward	

attnLRP = {
    Qwen2MLP: partial(patch_method, gated_mlp_forward),
    Qwen2_5_VLMLP: partial(patch_method, gated_mlp_forward),
    Qwen2RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen2_5_vl: patch_attention,
    GELU: partial(patch_method, non_linear_forward, keep_original=True),
}

cp_LRP = {
    Qwen2MLP: partial(patch_method, cp_gated_mlp_forward),
    Qwen2_5_VLMLP: partial(patch_method, cp_gated_mlp_forward),
    Qwen2RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen2_5_vl: patch_cp_attention,
    GELU: partial(patch_method, non_linear_forward, keep_original=True),
}



