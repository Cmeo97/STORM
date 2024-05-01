from .dino_transformer_utils import *
from math import sqrt
from typing import List, Optional, Union
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange

@dataclass
class TokenizerWithSAMEncoderOutput:
    z: torch.FloatTensor
    inits: torch.FloatTensor
    z_vit: torch.FloatTensor


@dataclass
class SAConfig:
    num_slots: int
    tokens_per_slot: int
    iters: int
    channels_enc: int
    token_dim: int
    prior_class: str

    @property
    def slot_dim(self):
        return self.tokens_per_slot * self.token_dim


class SlotAttentionVideo(nn.Module):
    def __init__(self, config: SAConfig, eps=1e-8, hidden_dim=128) -> None:
        super().__init__()

        self.slot_dim = config['tokens_per_slot'] * config['token_dim']
        assert self.slot_dim % config['tokens_per_slot'] == 0
        self.config = config
        self.num_slots = config['num_slots']
        self.tokens_per_slot = config['tokens_per_slot']
        self.iters = config['iters']
        self.eps = eps
        self.scale = self.slot_dim**-0.5

        self.slots_mu = nn.Parameter(torch.rand(1, 1, self.slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + self.slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(config['channels_enc'], self.slot_dim, bias=False)
        self.to_v = nn.Linear(config['channels_enc'], self.slot_dim, bias=False)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        hidden_dim = max(self.slot_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(self.slot_dim*4, self.slot_dim),
        )

        if config['prior_class'].lower() == 'mlp':
            self.prior = nn.Sequential(
                nn.Linear(self.slot_dim, self.slot_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.slot_dim, self.slot_dim),
            )
        elif config['prior_class'].lower() == 'gru':
            self.prior = nn.GRU(self.slot_dim, self.slot_dim, batch_first=True)
        elif config['prior_class'].lower() == 'grucell':
            self.prior = nn.GRUCell(self.slot_dim, self.slot_dim)
        elif config['prior_class'].lower() == 'none' or config['prior_class'].lower() == 'tf':
            self.prior = None
        else:
            raise NotImplementedError("prior class not implemented")
        
        self.prior_class = config['prior_class']
        self.pred_prior_from = config['pred_prior_from']
        self.predictor = TransformerEncoder(num_blocks=1, d_model=self.slot_dim, num_heads=4, dropout=0.1)
        
        self.norm_input = nn.LayerNorm(config['channels_enc'])
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_pre_ff = nn.LayerNorm(self.slot_dim)
        self.slot_dim = self.slot_dim
        
        self._init_params()

        self.is_video = True

    def _init_params(self):
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                torch.nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)
        torch.nn.init.zeros_(self.gru.bias_ih)
        torch.nn.init.zeros_(self.gru.bias_hh)
        torch.nn.init.orthogonal_(self.gru.weight_hh)

    def forward(self, inputs: torch.Tensor, num_slots: Optional[int] = None) -> torch.Tensor:
        assert len(inputs.shape) == 4
        b, T, n, d = inputs.shape
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k *= self.scale

        slots_list = []
        dots_list = []
        hidden = None
        for t in range(T):
            slots_init = slots

            for i in range(self.iters):
                slots_prev = slots

                slots = self.norm_slots(slots)
                q = self.to_q(slots)
                dots = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn = dots.softmax(dim=-1) + self.eps

                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])

                slots = self.gru(updates.view(-1, self.slot_dim),
                                 slots_prev.view(-1, self.slot_dim))
                slots = slots.view(-1, self.num_slots, self.slot_dim)

                # use MLP only when more than one iterations
                if i < self.iters - 1:
                    slots = slots + self.mlp(self.norm_pre_ff(slots))
                
            slots_list += [slots]
            dots_list += [dots.softmax(dim=-1)]
            
            # if prior_class = tf and pred_prior_from = last, STEVE
            slots_to_predict_from = slots_init if self.pred_prior_from.lower() == 'init' else slots
            if self.prior_class.lower() == 'mlp':
                slots = self.prior(slots_to_predict_from)
            elif self.prior_class.lower() == 'gru':
                slots, hidden = self.prior(slots_to_predict_from, hidden)
            elif self.prior_class.lower() == 'grucell':
                slots = self.prior(slots_to_predict_from.view(-1, self.slot_dim), hidden)
                slots = slots.view(-1, self.num_slots, self.slot_dim)
            elif self.prior_class.lower() == 'tf':
                slots = self.predictor(slots_to_predict_from)
            elif self.prior_class.lower() == 'none':
                pass

        slots_list = torch.stack(slots_list, dim=1)   # B, T, num_slots, slot_size
        dots_list = torch.stack(dots_list, dim=1)

        return slots_list, dots_list
    

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


class DinoSAM_OCextractor(nn.Module):
    def __init__(self, config: dict, decoder_config: dict, slot_attn_config: dict):
        super().__init__()
        
        self.decoder = MLPDecoder(decoder_config)
        self.slot_attn = SlotAttentionVideo(slot_attn_config)
        self.vit_model_name = config.vit_model_name
        self.vit_use_pretrained = config.vit_use_pretrained
        self.vit_freeze = config.vit_freeze
        self.vit_feature_level = config.vit_feature_level

        self._init_vit()
        self._init_pos_embed(decoder_config.dec_output_dim, slot_attn_config.token_dim)

        self.width = decoder_config.resolution
        self.height = decoder_config.resolution
        self.num_slots = slot_attn_config.num_slots
        self.tokens_per_slot = slot_attn_config.tokens_per_slot
        self.slot_based = True

    def __repr__(self) -> str:
        return "tokenizer"
    
    def _init_vit(self):
        def feature_level_to_list(feature_level):
            if feature_level is None:
                return []
            elif isinstance(feature_level, (int, str)):
                return [feature_level]
            else:
                return list(feature_level)

        self.feature_levels = feature_level_to_list(self.vit_feature_level)

        model = timm.create_model(self.vit_model_name, pretrained=self.vit_use_pretrained)
        # Delete unused parameters from classification head
        if hasattr(model, "head"):
            del model.head
        if hasattr(model, "fc_norm"):
            del model.fc_norm

        if len(self.feature_levels) > 0:
            self._feature_hooks = [
                _VitFeatureHook.create_hook_from_feature_level(level).register_with(model) for level in self.feature_levels
            ]
            feature_dim = model.num_features * len(self.feature_levels)

            # Remove modules not needed in computation of features
            max_block = max(hook.block for hook in self._feature_hooks)
            new_blocks = model.blocks[:max_block]  # Creates a copy
            del model.blocks
            model.blocks = new_blocks
            model.norm = nn.Identity()

        self.vit = model
        print(self.vit)
        self._feature_dim = feature_dim

        if self.vit_freeze:
            self.vit.requires_grad_(False)
            # BatchNorm layers update their statistics in train mode. This is probably not desired
            # when the model is supposed to be frozen.
            contains_bn = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                for m in self.vit.modules()
            )
            self.run_in_eval_mode = contains_bn
        else:
            self.run_in_eval_mode = False

    def _init_pos_embed(self, encoder_output_dim, token_dim):
        layers = []
        layers.append(nn.LayerNorm(encoder_output_dim))
        layers.append(nn.Linear(encoder_output_dim, encoder_output_dim))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(encoder_output_dim, token_dim))
        nn.init.zeros_(layers[-1].bias)
        self.pos_embed = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.inits, outputs.z_vit, reconstructions
  
    def cosine_loss(self, z: torch.Tensor):
        cosine_loss = 0.
        B, T, _, num_slots = z.shape
        for t in range(z.shape[1]-1):
            z_curr = z[:, t]
            z_next = z[:, t+1]
            z_curr = z_curr / z_curr.norm(dim=-2, keepdim=True)
            z_next = z_next / z_next.norm(dim=-2, keepdim=True)
            # matrix of cosine similarities between all pairs of slots for each sample in batch
            pairwise_sim = torch.bmm(z_curr.transpose(1, 2), z_next)
            loss = torch.pow(torch.diagonal(pairwise_sim, dim1=-2, dim2=-1) - torch.ones(num_slots, device=pairwise_sim.device).expand(B, -1), 2).mean()
            cosine_loss += loss
        cosine_loss /= (T-1)
        return cosine_loss

    def vit_encode(self, x: torch.Tensor) -> torch.Tensor:
        def _transformer_compute_positions(features):
            """Compute positions for Transformer features."""
            n_tokens = features.shape[1]
            image_size = sqrt(n_tokens)
            image_size_int = int(image_size)
            assert (
                image_size_int == image_size
            ), "Position computation for Transformers requires square image"
    
            spatial_dims = (image_size_int, image_size_int)
            positions = torch.cartesian_prod(
                *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
            )
            return positions
        
        if self.run_in_eval_mode and self.training:
            self.eval()
  
        if self.vit_freeze:
            print(f"Shape: {x.shape}")
            # Speed things up a nvidiabit by not requiring grad computation.
            with torch.no_grad():
                features = self.vit.forward_features(x)
        else:
            features = self.vit.forward_features(x)
  
        if self._feature_hooks is not None:
            hook_features = [hook.pop() for hook in self._feature_hooks]
  
        if len(self.feature_levels) == 0:
            # Remove class token when not using hooks.
            features = features[:, 1:]
            positions = _transformer_compute_positions(features)
        else:
            features = hook_features[: len(self.feature_levels)]
            positions = _transformer_compute_positions(features[0])
            features = torch.cat(features, dim=-1)
  
        return features, positions
    
    def dino_encode(self, x: torch.Tensor):
        x = self.preprocess_input(x)

        shape = x.shape  # (..., C, H, W)
        x = x.reshape(-1, *shape[-3:])
        z, _ = self.vit_encode(x)
        z = self.pos_embed(z)
  
        if self.slot_attn.is_video:
            z = z.reshape(*shape[:-3], *z.shape[1:]) # video
            if z.dim() == 3:
                z = z.unsqueeze(1)
        z, attns = self.slot_attn(z)
        z_vit, _ = self.vit_encode(x)
        z_vit = z_vit.reshape(*shape[:-3], *z_vit.shape[1:])
  
        return z, attns, z_vit

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape  # (..., C, D)
        z = z.view(-1, *shape[-2:])
        rec, _, _ = self.decoder(z)
        rec = rec.reshape(*shape[:-2], *rec.shape[1:])
        return rec
    
    def decode_slots(self, z: torch.Tensor, x: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z.shape  # (..., C, D)
        z = z.view(-1, *shape[-2:])
        rec, masks, masks_as_image = self.decoder(z)
        rec = rec.reshape(*shape[:-2], *rec.shape[1:])
        masks = masks.reshape(*shape[:-2], *masks.shape[1:])
        masks_as_image = masks_as_image.reshape(*shape[:-2], *masks_as_image.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
  
        colors = x.unsqueeze(-4).expand(-1, self.num_slots, -1, -1, -1) if len(x.shape) == 4 else x.unsqueeze(-4).expand(-1, -1, self.num_slots, -1, -1, -1)
      
        return x, colors, masks_as_image.unsqueeze(-3)
    
    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        # z_q = self.encode(x, should_preprocess).z_quantized
        # return self.decode(z_q, should_postprocess)
        return x
    
    @torch.no_grad()
    def encode_decode_slots(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False, use_hard: bool = False) -> torch.Tensor:
        z = self.encode(x, should_preprocess).z
        return self.decode_slots(z, x, should_postprocess)
  
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)
  
    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
    

class PositionalEmbedding(nn.Module):
    def __init__(self, resolution: List[int], channels: int):
        super().__init__()
        height, width = resolution
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, resolution: int, dec_input_dim: int, dec_hidden_dim: int, out_ch: int) -> None:
        super().__init__()
        
        hidden_dim = dec_hidden_dim
        out_resolution = resolution

        if hidden_dim == 64:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(dec_input_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                # nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim, out_ch, 3, stride=(1, 1), padding=1),
            )
        elif hidden_dim == 32:
            self.layers = nn.Sequential(
                # nn.ConvTranspose2d(config.dec_input_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.Conv2d(dec_input_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.Conv2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.Conv2d(hidden_dim, hidden_dim, 5, stride=(1, 1), padding=2),
                nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(hidden_dim, config.out_ch, 3, stride=(1, 1), padding=1),
                nn.Conv2d(hidden_dim, out_ch, 3, stride=(1, 1), padding=1),
            )
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.init_resolution = (28, 28) if out_resolution == 224 else (8, 8)
        #self.init_resolution = resolution if hidden_dim == 32 else (28, 28)
        self.pos_embedding = PositionalEmbedding(self.init_resolution, dec_input_dim)
        self.resolution = resolution
        self._init_params()
        print('Decoder initialized')

    def _init_params(self):
        for name, tensor in self.named_parameters():
            if name.endswith(".bias"):
                nn.init.zeros_(tensor)
            elif len(tensor.shape) <= 1:
                pass  # silent
            else:
                nn.init.xavier_uniform_(tensor)

    def __repr__(self) -> str:
        return "image_decoder"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        K = x.shape[1] # number of slots
        x = self.spatial_broadcast(x)
        x = self.pos_embedding(x)
        x = self.layers(x)

        # Undo combination of slot and batch dimension; split alpha masks.
        colors, masks = x[:, :3], x[:, -1:]
        colors = colors.reshape(bs, K, 3, self.resolution[0], self.resolution[1])
        masks = masks.reshape(bs, K, 1, self.resolution[0], self.resolution[1])
        masks = masks.softmax(dim=1)
        rec = (colors * masks).sum(dim=1)

        return rec, colors, masks

    def spatial_broadcast(self, slot: torch.Tensor) -> torch.Tensor:
        slot = slot.reshape(-1, slot.shape[-1])
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.init_resolution[0], self.init_resolution[1])
    
    
    
  
  