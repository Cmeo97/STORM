import enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt
import timm




def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    
    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        if attn_mask is not None:
            if attn.dtype == torch.float16:
                attn = attn.masked_fill(attn_mask == 0, -6e4)
            else:
                attn = attn.masked_fill(attn_mask == 0, -1e9)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output
    
class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()
        
        self.is_first = is_first
        
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        if self.is_first:
            input = self.attn_layer_norm(input)
            x = self.attn(input, input, input)
            input = input + x
        else:
            x = self.attn_layer_norm(input)
            x = self.attn(x, x, x)
            input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerEncoder(nn.Module):    
    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()
        
        if num_blocks > 0:
            gain = (2 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        for block in self.blocks:
            input = block(input)
        
        return self.layer_norm(input)


class _VitFeatureType(enum.Enum):
    BLOCK = 1
    KEY = 2
    VALUE = 3
    QUERY = 4
    CLS = 5


class _VitFeatureHook:
    """Auxilliary class used to extract features from timm ViT models."""

    def __init__(self, feature_type: _VitFeatureType, block: int, drop_cls_token: bool = True):
        """Initialize VitFeatureHook.

        Args:
            feature_type: Type of feature to extract.
            block: Number of block to extract features from. Note that this is not zero-indexed.
            drop_cls_token: Drop the cls token from the features. This assumes the cls token to
                be the first token of the sequence.
        """
        assert isinstance(feature_type, _VitFeatureType)
        self.feature_type = feature_type
        self.block = block
        self.drop_cls_token = drop_cls_token
        self.name = f"{feature_type.name.lower()}{block}"
        self.remove_handle = None  # Can be used to remove this hook from the model again

        self._features = None

    @staticmethod
    def create_hook_from_feature_level(feature_level: Union[int, str]):
        feature_level = str(feature_level)
        prefixes = ("key", "query", "value", "block", "cls")
        for prefix in prefixes:
            if feature_level.startswith(prefix):
                _, _, block = feature_level.partition(prefix)
                feature_type = _VitFeatureType[prefix.upper()]
                block = int(block)
                break
        else:
            feature_type = _VitFeatureType.BLOCK
            try:
                block = int(feature_level)
            except ValueError:
                raise ValueError(f"Can not interpret feature_level '{feature_level}'.")

        return _VitFeatureHook(feature_type, block)

    def register_with(self, model):
        supported_models = (
            timm.models.vision_transformer.VisionTransformer,
            timm.models.beit.Beit,
            timm.models.vision_transformer_sam.VisionTransformerSAM,
        )
        model_names = ["vit", "beit", "samvit"]

        if not isinstance(model, supported_models):
            raise ValueError(
                f"This hook only supports classes {', '.join(str(cl) for cl in supported_models)}."
            )

        if self.block > len(model.blocks):
            raise ValueError(
                f"Trying to extract features of block {self.block}, but model only has "
                f"{len(model.blocks)} blocks"
            )

        block = model.blocks[self.block - 1]
        if self.feature_type == _VitFeatureType.BLOCK:
            self.remove_handle = block.register_forward_hook(self)
        else:
            if isinstance(block, timm.models.vision_transformer.ParallelBlock):
                raise ValueError(
                    f"ViT with `ParallelBlock` not supported for {self.feature_type} extraction."
                )
            elif isinstance(model, timm.models.beit.Beit):
                raise ValueError(f"BEIT not supported for {self.feature_type} extraction.")
            self.remove_handle = block.attn.qkv.register_forward_hook(self)

        model_name_map = dict(zip(supported_models, model_names))
        self.model_name = model_name_map.get(type(model), None)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp):
        if self.feature_type == _VitFeatureType.BLOCK:
            features = outp
            if self.drop_cls_token:
                # First token is CLS token.
                if self.model_name == "samvit":
                    # reshape outp (B,H,W,C) -> (B,H*W,C)
                    features = outp.flatten(1,2)
                else:
                    features = features[:, 1:]
        elif self.feature_type in {
            _VitFeatureType.KEY,
            _VitFeatureType.QUERY,
            _VitFeatureType.VALUE,
        }:
            # This part is adapted from the timm implementation. Unfortunately, there is no more
            # elegant way to access keys, values, or queries.
            B, N, C = inp[0].shape
            qkv = outp.reshape(B, N, 3, C)  # outp has shape B, N, 3 * H * (C // H)
            q, k, v = qkv.unbind(2)

            if self.feature_type == _VitFeatureType.QUERY:
                features = q
            elif self.feature_type == _VitFeatureType.KEY:
                features = k
            else:
                features = v
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type == _VitFeatureType.CLS:
            # We ignore self.drop_cls_token in this case as it doesn't make any sense.
            features = outp[:, 0]  # Only get class token.
        else:
            raise ValueError("Invalid VitFeatureType provided.")

        self._features = features


def resize_patches_to_image(patches: torch.Tensor, size: Optional[int] = None, 
                            scale_factor: Optional[float] = None, resize_mode: str = "bilinear") -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = sqrt(n_patches)
    patch_size = int(sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


class MLPDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.dec_input_dim = config.dec_input_dim
        self.dec_hidden_layers = config.dec_hidden_layers
        self.dec_output_dim = config.dec_output_dim

        self.vit_num_patches = config.vit_num_patches
        
        layers = []
        current_dim = config.dec_input_dim
    
        for dec_hidden_dim in config.dec_hidden_layers:
            layers.append(nn.Linear(current_dim, dec_hidden_dim))
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU(inplace=True))
            current_dim = dec_hidden_dim

        layers.append(nn.Linear(current_dim, config.dec_output_dim + 1))
        nn.init.zeros_(layers[-1].bias)
        
        self.layers = nn.Sequential(*layers)

        self.pos_embed = nn.Parameter(torch.randn(1, config.vit_num_patches, config.dec_input_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z (bt, k, d)
        init_shape = z.shape[:-1]
        z = z.flatten(0, -2)
        z = z.unsqueeze(1).expand(-1, self.vit_num_patches, -1)

        # Simple learned additive embedding as in ViT
        z = z + self.pos_embed
        out = self.layers(z)
        out = out.unflatten(0, init_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.dec_output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        masks_as_image = resize_patches_to_image(masks, size=self.config.resolution, resize_mode="bilinear")

        return reconstruction, masks, masks_as_image

class BroadcastPoolLayer(nn.Module):
    def __init__(self, dec_input_dim, dec_hidden_layers, dec_output_dim) -> None:
        super().__init__()

   
        self.dec_input_dim = dec_input_dim
        self.dec_hidden_layers = dec_hidden_layers
        self.dec_output_dim = dec_output_dim
        
        layers = []
        current_dim = dec_input_dim
    
        for dec_hidden_dim in dec_hidden_layers:
            layers.append(nn.Linear(current_dim, dec_hidden_dim))
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU(inplace=True))
            current_dim = dec_hidden_dim

        layers.append(nn.Linear(current_dim, dec_output_dim + 1))
        nn.init.zeros_(layers[-1].bias)
        
        self.layers = nn.Sequential(*layers)

        self.pos_embed = nn.Parameter(torch.randn(1, 1, 1, dec_input_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
    
        z = z + self.pos_embed
        out = self.layers(z)
       
        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.dec_output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-2)
        pooled_z = torch.sum(decoded_patches * alpha, dim=-2)
        
        return pooled_z