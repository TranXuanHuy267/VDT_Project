import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F
import torch

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class ParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,  # NOTE: not used
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = 1 # use_fused_attn() # boolean
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x