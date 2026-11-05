import types
from functools import partial

import torch
from timm.layers import maybe_add_mask, resolve_self_attn_mask
from timm.layers.attention import Attention
from timm.layers.norm import LayerNorm as TimmLayerNorm

from lxt.efficient.patches import (
    dropout_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_method,
    stop_gradient,
)

def install_conservative_pos_embed(model, eps=1e-6):
    """
    Apply before monkey-pathcing.
    Replace timm ViT._pos_embed and expose stable constant accounting.

    Positional embeddings use ordinary addition. This intentionally leaves their
    relevance measurable instead of using an unstable component-wise division.
    The eps argument is retained for command-line compatibility but is unused.
    """

    def _pos_embed_lrp(self, x):
        if self.pos_embed is None:
            return x

        if self.dynamic_img_size:
            raise NotImplementedError(
                "This example targets fixed-size ViTs. Add timm's resample_abs_pos_embed "
                "before conservative_add_constant for dynamic image sizes."
            )

        batch_size = x.shape[0]
        prefix = []
        if self.cls_token is not None:
            prefix.append(self.cls_token.expand(batch_size, -1, -1))
        if getattr(self, "reg_token", None) is not None:
            prefix.append(self.reg_token.expand(batch_size, -1, -1))

        if self.no_embed_class:
            # In this timm mode, positional embeddings belong only to patch tokens.
            x = x + self.pos_embed
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
        else:
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
            x = x + self.pos_embed

        return self.pos_drop(x)

    model._pos_embed = types.MethodType(_pos_embed_lrp, model)


def timm_attention_forward(self, x, attn_mask=None, is_causal=False):
    batch_size, num_tokens, channels = x.shape
    qkv = (
        self.qkv(x)
        .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    # CP-LRP inside attention: treat the attention weights as contextual
    # constants and propagate relevance through the value path. LXT recommends
    # this hybrid for ViTs because it is substantially easier to stabilize with
    # Gamma rules than the full Q/K/V relevance split.
    q = stop_gradient(q)
    k = stop_gradient(k)

    if self.fused_attn:
        raise RuntimeError("Set block.attn.fused_attn = False before attribution.")

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn_bias = resolve_self_attn_mask(num_tokens, attn, attn_mask, is_causal)
    attn = maybe_add_mask(attn, attn_bias)
    attn = self.attn_drop(attn.softmax(dim=-1))
    x = attn @ v

    x = x.transpose(1, 2).reshape(batch_size, num_tokens, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    return self.proj_drop(x)


def identity_forward(self, x):
    return x


attnLRP = {
    torch.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    torch.nn.Dropout: partial(patch_method, dropout_forward),
    # timm uses its own LayerNorm class; an exact class-key patch is required.
    TimmLayerNorm: partial(patch_method, layer_norm_forward),
    Attention: partial(
        patch_method, timm_attention_forward, keep_original=True
    ),
    torch.nn.Identity: partial(patch_method, identity_forward, keep_original=True),
}

