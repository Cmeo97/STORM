from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from torchvision.transforms import Resize
from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask, get_causal_mask
from sub_models.transformer_model import StochasticTransformerKVCache, TransformerXL
from agents import TransformerWithCLS
import agents
from math import sqrt
from sub_models.dino_sam import DinoSAM_OCextractor, SpatialBroadcastDecoder
from typing import Union, Iterable, List, Dict, Tuple, Optional
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def linear_warmup_exp_decay(warmup_steps: Optional[int] = None, exp_decay_rate: Optional[float] = None, exp_decay_steps: Optional[int] = None):
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0
    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier
    return lr_lambda

class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        feature_width = 64//2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels*final_feature_width*final_feature_width, bias=False))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_channels, H=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        #self.apply(init_weights)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs + 1e-6)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L S (K C) -> B L S K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x, generation=False):
        logits = self.prior_head(x)
        if generation:
            logits = rearrange(logits, "B L S (K C) -> B L S K C", K=self.stoch_dim)
        else:
            if logits.dim() == 4:
                logits = rearrange(logits, "B L S (K C) -> B L S K C", K=self.stoch_dim)
            else:
                logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)

        logits = self.unimix(logits)
        return logits

class OCDistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)

    def forward_post(self, x):
        logits = self.post_head(x)
        return logits

    def forward_prior(self, x, generation=False):
        logits = self.prior_head(x)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
 
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        #self.apply(init_weights)



    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        #self.apply(init_weights)

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination
    
class SlotsHead(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, embedding_size),
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        #self.apply(init_weights)

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim, transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, conf):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_dim = conf.Models.WorldModel.stochastic_dim
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        dtype = conf.BasicSettings.dtype
        self.use_amp = True if (dtype == 'torch.float16' or dtype == 'torch.bfloat16') else False
        self.tensor_dtype = torch.float16 if dtype == 'torch.float16' else torch.bfloat16 if dtype == 'torch.bfloat16' else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.conf = conf
        self.num_slots = self.conf.Models.Slot_attn.num_slots
        self.slots_size = self.conf.Models.Slot_attn.token_dim

        if conf.Models.WorldModel.model == 'OC-irisXL':
            decoder_config = conf.Models.Decoder
            slot_attn_config = conf.Models.Slot_attn
            self.dino = DinoSAM_OCextractor(conf.Models.WorldModel, decoder_config, slot_attn_config)
            
            self.storm_transformer = TransformerXL(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=transformer_hidden_dim,
                transformer_layer_config=conf.Models.WorldModel.transformer_layer, 
                num_layers=conf.Models.WorldModel.TransformerNumLayers,
                max_length=conf.Models.WorldModel.TransformerMaxLength,
                mem_length=3*conf.Models.Slot_attn.num_slots,
                conf=conf
            )
            self.image_decoder = SpatialBroadcastDecoder(
                resolution=conf.Models.Decoder.resolution, 
                dec_input_dim=conf.Models.Decoder.dec_input_dim,
                dec_hidden_dim=conf.Models.Decoder.dec_hidden_dim,
                out_ch=conf.Models.Decoder.out_ch
            )
            if conf.Models.WorldModel.stochastic_head:
                self.dist_head = DistHead(
                    image_feat_dim=conf.Models.Slot_attn.token_dim,
                    transformer_hidden_dim=transformer_hidden_dim,
                    stoch_dim=self.stoch_dim
                )
            else:
                self.dist_head = OCDistHead(
                    image_feat_dim=conf.Models.Slot_attn.token_dim,
                    transformer_hidden_dim=transformer_hidden_dim,
                    stoch_dim=self.stoch_dim
                )
            
            if conf.Models.WorldModel.wm_oc_pool_layer == 'cls-transformer':
                self.wm_oc_pool_layer = TransformerWithCLS(transformer_hidden_dim, transformer_hidden_dim, self.conf.Models.CLSTransformer.NumHeads, self.conf.Models.CLSTransformer.NumLayers)
            else:
                self.wm_oc_pool_layer = nn.Sequential(
                    nn.Linear(self.conf.Models.Slot_attn.num_slots*transformer_hidden_dim, transformer_hidden_dim, bias=False),
                    nn.LayerNorm(transformer_hidden_dim),
                    nn.ReLU()
                    )
                
            self.slots_head = SlotsHead(
                embedding_size=conf.Models.Slot_attn.token_dim,
                transformer_hidden_dim=transformer_hidden_dim
            )
            self.reward_decoder = RewardDecoder(
                num_classes=255,
                embedding_size=transformer_hidden_dim,
                transformer_hidden_dim=transformer_hidden_dim
            )
            self.termination_decoder = TerminationDecoder(
                embedding_size=transformer_hidden_dim,
                transformer_hidden_dim=transformer_hidden_dim
            )
            
            self.downsample = Resize(size=(conf.Models.Decoder.resolution, conf.Models.Decoder.resolution))
            self.dino_parameters = list(self.dino.parameters()) 
            self.wm_parameters = list(self.storm_transformer.parameters()) + list(self.wm_oc_pool_layer.parameters()) + list(self.slots_head.parameters()) + list(self.termination_decoder.parameters()) + list(self.reward_decoder.parameters()) + list(self.dist_head.parameters()) 
            self.dec_parameters = list(self.image_decoder.parameters())
            self.dino_optimizer = torch.optim.Adam(self.dino_parameters, lr=0.0002)
            self.wm_optimizer = torch.optim.Adam(self.wm_parameters, lr=1e-4)
            self.dec_optimizer = torch.optim.Adam(self.dec_parameters, lr=0.0002)

            self.dino_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dino_optimizer, lr_lambda=linear_warmup_exp_decay(10000, 0.5, 100000))
            self.wm_scheduler = None
            self.dec_scheduler = None

        else:
            self.encoder = EncoderBN(
                in_channels=in_channels,
                stem_channels=32,
                final_feature_width=self.final_feature_width
            )
            self.storm_transformer = StochasticTransformerKVCache(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=transformer_hidden_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                max_length=transformer_max_length,
                dropout=0.1
            )
            self.image_decoder = DecoderBN(
                stoch_dim=self.stoch_flattened_dim,
                last_channels=self.encoder.last_channels,
                original_in_channels=in_channels,
                stem_channels=32,
                final_feature_width=self.final_feature_width
            )
            self.dist_head = DistHead(
                image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
                transformer_hidden_dim=transformer_hidden_dim,
                stoch_dim=self.stoch_dim
            )

            self.reward_decoder = RewardDecoder(
                num_classes=255,
                embedding_size=self.stoch_flattened_dim,
                transformer_hidden_dim=transformer_hidden_dim
            )
            self.termination_decoder = TerminationDecoder(
                embedding_size=self.stoch_flattened_dim,
                transformer_hidden_dim=transformer_hidden_dim
            )
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)


        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def load(self, path, device):
        def extract_state_dict(state_dict, module_name):
            return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})
        
        self.dino.load_state_dict(extract_state_dict(torch.load(path, map_location=device), 'tokenizer'), strict=False)
        self.image_decoder.load_state_dict(extract_state_dict(torch.load(path, map_location=device), 'image_decoder'))

    def encode_obs(self, obs):
        if self.conf.Models.WorldModel.model=='OC-irisXL':
            with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
                slots, attns, z_vit = self.dino.dino_encode(obs) 
                post_logits = self.dist_head.forward_post(slots) 
                sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample") if self.conf.Models.WorldModel.stochastic_head else post_logits
                return slots, sample, attns, z_vit
        else:
            with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
                embedding = self.encoder(obs)
                post_logits = self.dist_head.forward_post(embedding)
                sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
                flattened_sample = self.flatten_sample(sample)
            return flattened_sample
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape  # (..., C, D)
        z = z.view(-1, *shape[-3:])
        rec = self.image_decoder(z)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        return rec

    def calc_last_dist_feat(self, latent, action, termination=None, mems=None, device='cuda:0'):
        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            if isinstance(self.storm_transformer, StochasticTransformerKVCache):
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.storm_transformer(latent, action, temporal_mask)
                prior_logits = self.dist_head.forward_prior(dist_feat[:, -1:]) 
                prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
                prior_flattened_sample = self.flatten_sample(prior_sample)
                return prior_flattened_sample, dist_feat[:, -1:]
            else:
                src_length = tgt_length = latent.shape[1] * self.conf.Models.Slot_attn.num_slots
                src_length = src_length + mems[0].shape[0] if mems is not None else src_length
                sequence_length = latent.shape[1] + mems[0].shape[0]/self.conf.Models.Slot_attn.num_slots 
                positions = torch.arange(sequence_length -1, -1, -1, device=device).repeat_interleave(self.conf.Models.Slot_attn.num_slots, dim=0).long() if self.conf.Models.WorldModel.slot_based  else torch.arange(src_length - 1, -1, -1, device=device) 
                temporal_mask = get_causal_mask(src_length, tgt_length, device, torch.tensor(termination).t().to(device), self.conf.Models.Slot_attn.num_slots, mem_num_tokens=mems[0].shape[0], generation=True)
                if latent.dim() == 5:
                    latent = rearrange(latent, 'b t s e E->b t s (e E)')
                dist_feat, mems = self.storm_transformer(latent, action, temporal_mask, positions, mems, generation=True)
                prior_logits = self.dist_head.forward_prior(dist_feat[:, -1:], generation=True) if self.conf.Models.WorldModel.stochastic_head else self.slots_head(dist_feat[:, -1:])
                prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample") if self.conf.Models.WorldModel.stochastic_head else prior_logits
                prior_flattened_sample = self.flatten_sample(prior_sample) if self.conf.Models.WorldModel.stochastic_head else prior_sample
                return prior_flattened_sample, dist_feat[:, -1:], mems

    def predict_next(self, last_flattened_sample, action, termination, log_video=True, mems=None, device=None):
        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            if isinstance(self.storm_transformer, StochasticTransformerKVCache):
                dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
            else:
                src_length = last_flattened_sample.shape[1] * self.conf.Models.Slot_attn.num_slots
                src_length = src_length + mems[0].shape[0] if mems is not None else src_length
                sequence_length = last_flattened_sample.shape[1] + mems[0].shape[0]/self.conf.Models.Slot_attn.num_slots if mems is not None else last_flattened_sample.shape[1]
                positions = torch.arange(sequence_length -1, -1, -1, device=device).repeat_interleave(self.conf.Models.Slot_attn.num_slots, dim=0).long() if self.conf.Models.WorldModel.slot_based  else torch.arange(src_length - 1, -1, -1, device=device) 
                temporal_mask = get_causal_mask(src_length, self.conf.Models.Slot_attn.num_slots, device, termination, self.conf.Models.Slot_attn.num_slots, mem_num_tokens=mems[0].shape[0], generation=True)
                if last_flattened_sample.dim() == 5:
                    last_flattened_sample = rearrange(last_flattened_sample, 'b t s e E->b t s (e E)')
                
                dist_feat, mems = self.storm_transformer(last_flattened_sample, action, temporal_mask, positions, mems, generation=True)
            

            # decoding
            if self.conf.Models.WorldModel.stochastic_head:
                prior_logits = self.dist_head.forward_prior(dist_feat, generation=True)
                prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
                prior_flattened_sample = self.flatten_sample(prior_sample)
            else:
                if self.conf.Models.WorldModel.wm_oc_pool_layer == 'dino-sbd' or self.conf.Models.WorldModel.wm_oc_pool_layer == 'mlp':
                    slots_hat = self.slots_head(dist_feat)
                    prior_flattened_sample = slots_hat
                else:
                    prior_flattened_sample = dist_feat
        
            if log_video:
                slots_hat = self.slots_head(dist_feat[:8])
                seq_len = slots_hat.shape[1]
                slots_hat = rearrange(slots_hat, 'b t s e -> (b t) s e')
                recon, colors, masks = self.image_decoder(slots_hat)
                recon = rearrange(recon, '(b t) c h w -> b t c h w', t=seq_len)
                colors = rearrange(colors, '(b t) k c h w -> b t k c h w', t=seq_len)
                masks = rearrange(masks, '(b t) k c h w -> b t k c h w', t=seq_len)
                output_hat = recon, colors, masks
            else:
                output_hat = None
            
            if self.conf.Models.WorldModel.model=='OC-irisXL':
                combined_dist_feat = rearrange(dist_feat, 'b t s e-> b t (s e)') if self.conf.Models.WorldModel.wm_oc_pool_layer != 'cls-transformer' else dist_feat
                combined_dist_feat = self.wm_oc_pool_layer(combined_dist_feat)
                reward_hat = self.reward_decoder(combined_dist_feat)
                reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
                termination_hat = self.termination_decoder(combined_dist_feat)
                termination_hat = termination_hat > 0
            else:
                reward_hat = self.reward_decoder(dist_feat)
                reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
                termination_hat = self.termination_decoder(dist_feat)
                termination_hat = termination_hat > 0
  
        return output_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat, mems

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        if sample.dim() == 5:
            sample = rearrange(sample, "B L T K C -> B L T (K C)")
        else:
            sample = rearrange(sample, "B L K C -> B L (K C)")
        return sample

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            if self.conf.Models.WorldModel.model=='OC-irisXL':
                latent_size = (imagine_batch_size, imagine_batch_length+1, self.num_slots, self.slots_size)
                hidden_size = (imagine_batch_size, imagine_batch_length+1, self.num_slots, self.transformer_hidden_dim)
            else:
                latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
                hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action, sample_termination,
                     imagine_batch_size, imagine_batch_length, log_video, logger, device):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=device)
        obs_hat_list, colors_hat_list, masks_hat_list = [], [], []
        if log_video:
            sample_obs, obs_gt = torch.chunk(sample_obs, 2, 1)
       
        if isinstance(self.storm_transformer, StochasticTransformerKVCache):
            self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype, device=device)
        # context
        if self.conf.Models.WorldModel.model == 'OC-irisXL':
            context_slots, context_latent, _, _  = self.encode_obs(sample_obs)
            mems = self.storm_transformer.init_mems()
        else:
            context_latent = self.encode_obs(sample_obs)
            mems = None
        
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat, mems = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                sample_termination[:, i:i+1],
                log_video=log_video,
                mems=mems, 
                device=device
            )
        self.latent_buffer[:, 0:1] = last_latent   # change initialization of self.latent_buffer
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action = agent.sample((self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]))
            self.action_buffer[:, i:i+1] = action
            with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
                last_flattened_sample = self.dist_head.forward_post(self.latent_buffer[:, i:i+1])
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat, mems = self.predict_next(
                last_flattened_sample, self.action_buffer[:, i:i+1], self.termination_hat_buffer[:, i:i+1] ,log_video=log_video, mems=mems, device=device)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                last_obs_hat, last_colors_hat, last_masks_hat = last_obs_hat
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env
                colors_hat_list.append(last_colors_hat[::imagine_batch_size//16]) 
                masks_hat_list.append(last_masks_hat[::imagine_batch_size//16]) 
        
        if log_video:
            obs_hat = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1)
            obs_downsampled = self.downsample(rearrange(obs_gt[:obs_hat_list[0].shape[0]], 'b t c h w->(b t c) h w')).view(*obs_hat.shape)
            colors_hat, masks_hat = torch.cat(colors_hat_list, dim=1), torch.cat(masks_hat_list, dim=1)
            logger.log("Imagine/predict_video", obs_hat.cpu().float().detach().numpy())
            obs_hat_list = self.compute_image_with_slots(obs_downsampled, obs_hat, colors_hat, masks_hat)
            logger.log("Imagine/predict_slots_images", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).squeeze(1).cpu().float().detach().numpy(), imagine_batch_length)

        return (self.latent_buffer, self.hidden_buffer), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def update(self, obs, action, reward, termination, logger=None, log_recs=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            obs_downsampled = self.downsample(rearrange(obs, 'b t c h w -> (b t) c h w'))
            obs_downsampled = rearrange(obs_downsampled, '(b t) c h w -> b t c h w', b=batch_size)

            # encoding
            if self.conf.Models.WorldModel.model == 'OC-irisXL':
                embedding, attns, z_vit = self.dino.dino_encode(obs.reshape(-1, 4, *obs.shape[2:]))  # embedding = slots
                reconstructions = self.dino.decode(embedding)
                embedding = embedding.reshape(batch_size, batch_length, *embedding.shape[2:])
                history_length = embedding.shape[1]
                src_length = tgt_length = history_length * self.conf.Models.Slot_attn.num_slots
                device = embedding.device
                positions = torch.arange(history_length - 1, -1, -1, device=device).repeat_interleave(self.conf.Models.Slot_attn.num_slots, dim=0).long() if self.conf.Models.WorldModel.slot_based  else torch.arange(src_length - 1, -1, -1, device=device) 
            else:
                embedding = self.encoder(obs)
                
            post_logits = self.dist_head.forward_post(embedding.detach()) if self.conf.Models.WorldModel.independent_modules else self.dist_head.forward_post(embedding)
            if self.conf.Models.WorldModel.stochastic_head:
                sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            
            flattened_sample = self.flatten_sample(sample) if self.conf.Models.WorldModel.stochastic_head else post_logits
            
            # decoding image
            obs_hat, colors, masks = self.image_decoder(rearrange(embedding, 'b t s e -> (b t) s e').detach()) if self.conf.Models.WorldModel.model=='OC-irisXL' else self.image_decoder(flattened_sample)
            
            # transformer
            temporal_mask = get_causal_mask(src_length, tgt_length, embedding.device, termination, self.conf.Models.Slot_attn.num_slots) if self.conf.Models.WorldModel.model == 'OC-irisXL' else get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask, positions) if self.conf.Models.WorldModel.model == 'OC-irisXL' else self.storm_transformer(flattened_sample, action, temporal_mask) 
            if self.conf.Models.WorldModel.stochastic_head:
                prior_logits = self.dist_head.forward_prior(dist_feat)
            # decoding reward and termination with dist_feat
            if self.conf.Models.WorldModel.model=='OC-irisXL':
                slots_hat = self.slots_head(dist_feat)
                combined_dist_feat = rearrange(dist_feat, 'b t s e-> b t (s e)') if self.conf.Models.WorldModel.wm_oc_pool_layer != 'cls-transformer' else dist_feat
                combined_dist_feat = self.wm_oc_pool_layer(combined_dist_feat)
                reward_hat = self.reward_decoder(combined_dist_feat)
                termination_hat = self.termination_decoder(combined_dist_feat)
            else:
                reward_hat = self.reward_decoder(dist_feat)
                termination_hat = self.termination_decoder(dist_feat)

            # DINO losses
            consistency_loss = self.dino.cosine_loss(attns) if self.conf.Models.WorldModel.model == 'OC-irisXL' else 0
            dino_reconstruction_loss = torch.pow(z_vit - reconstructions, 2).mean() if self.conf.Models.WorldModel.model == 'OC-irisXL' else 0
            # decoder losses
            obs_hat = rearrange(obs_hat, '(b t) c h w -> b t c h w', b=batch_size)
            decoder_loss = torch.pow(obs_hat - obs_downsampled, 2).mean() if self.conf.Models.WorldModel.model == 'OC-irisXL' else 0
            dino_loss = dino_reconstruction_loss + consistency_loss if self.conf.Models.WorldModel.model == 'OC-irisXL' else self.mse_loss_func(obs_hat, obs)
            # STORM env losses
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            slots_loss = F.mse_loss(embedding.detach(), slots_hat)
            # dyn-rep loss
            
            wm_loss = reward_loss + slots_loss + termination_loss 
            total_loss = dino_loss + decoder_loss + wm_loss
            if self.conf.Models.WorldModel.stochastic_head:
                dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].reshape(-1,  *post_logits.shape[2:]).detach(), prior_logits.reshape(*post_logits.shape)[:, :-1].reshape(-1,  *post_logits.shape[2:]))
                representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].reshape(-1,  *post_logits.shape[2:]), prior_logits.reshape(*post_logits.shape)[:, :-1].reshape(-1,  *post_logits.shape[2:]).detach())
                total_loss += 0.5*dynamics_loss + 0.1*representation_loss

        # gradient descent
        if self.conf.Models.WorldModel.model=='OC-irisXL':
            # DINO Optimization
            self.scaler.scale(dino_loss).backward()
            self.scaler.unscale_(self.dino_optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.dino_parameters, max_norm=1.0)
            self.scaler.step(self.dino_optimizer)
            self.scaler.update()
            self.dino_optimizer.zero_grad(set_to_none=True)
            # DEC Optimization
            self.scaler.scale(decoder_loss).backward()
            self.scaler.unscale_(self.dec_optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.dec_parameters, max_norm=1.0)
            self.scaler.step(self.dec_optimizer)
            self.scaler.update()
            self.dec_optimizer.zero_grad(set_to_none=True)
            # WM Optimization
            self.scaler.scale(wm_loss).backward()
            self.scaler.unscale_(self.wm_optimizer)  # for clip grad
            self.scaler.unscale_(self.dec_optimizer)  # for clip grad
            dino_norm = torch.nn.utils.clip_grad_norm_(self.dino_parameters, max_norm=1.0)
            wm_norm = torch.nn.utils.clip_grad_norm_(self.wm_parameters, max_norm=10.0)
            dec_norm = torch.nn.utils.clip_grad_norm_(self.dec_parameters, max_norm=1.0)
            self.scaler.step(self.dino_optimizer)
            self.scaler.step(self.wm_optimizer)
            self.scaler.step(self.dec_optimizer)
            self.scaler.update()

            if self.dino_scheduler is not None:
                self.dino_scheduler.step()
            if self.wm_scheduler is not None:
                self.wm_scheduler.step()
            if self.dec_scheduler is not None:
                self.dec_scheduler.step()

            self.dino_optimizer.zero_grad(set_to_none=True)
            self.wm_optimizer.zero_grad(set_to_none=True)
            self.dec_optimizer.zero_grad(set_to_none=True)
        else:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/consistency_loss", consistency_loss.item())
            logger.log("WorldModel/dino_reconstruction_loss", dino_reconstruction_loss.item())
            logger.log("WorldModel/decoder_reconstruction_loss", decoder_loss.item())
            logger.log("WorldModel/dino_norm", dino_norm.item())
            logger.log("WorldModel/decoder_norm", dec_norm.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/slots_loss", slots_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            if self.conf.Models.WorldModel.stochastic_head:
                logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
                logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
                logger.log("WorldModel/representation_loss", representation_loss.item())
                logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
            logger.log("WorldModelNorm/dino_norm", dino_norm.item())
            logger.log("WorldModelNorm/wm_norm", wm_norm.item())
            logger.log("WorldModelNorm/dec_norm", dec_norm.item())
            if log_recs:
                #logger.log("WorldModel/predict_slots_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())
                obs_hat_list = self.compute_image_with_slots(obs_downsampled, obs_hat, rearrange(colors, '(b t) s c h w -> b t s c h w', b=batch_size), rearrange(masks, '(b t) s c h w -> b t s c h w', b=batch_size))
                logger.log("DINO/images/predict_slots_recs", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).squeeze(1).cpu().float().detach().numpy(), obs_hat.shape[1])
    

    def update_dino(self, obs, action, reward, termination, logger=None, log_recs=False):
        # self.train()

        ###########################
        ###########################
        # train dino
        self.eval()
        self.dino.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            embedding, attns, z_vit = self.dino.dino_encode(obs.reshape(-1, 4, *obs.shape[2:]))  # embedding = slots
            reconstructions = self.dino.decode(embedding)
            embedding = embedding.reshape(batch_size, batch_length, *embedding.shape[2:])
            consistency_loss = self.dino.cosine_loss(attns)
            dino_reconstruction_loss = torch.pow(z_vit - reconstructions, 2).mean()
            dino_loss = dino_reconstruction_loss + consistency_loss

        self.scaler.scale(dino_loss).backward()
        self.scaler.unscale_(self.dino_optimizer)  # for clip grad
        dino_norm = torch.nn.utils.clip_grad_norm_(self.dino_parameters, max_norm=1.0)
        self.scaler.step(self.dino_optimizer)
        self.scaler.update()

        if self.dino_scheduler is not None:
            self.dino_scheduler.step()

        self.dino_optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/consistency_loss", consistency_loss.item())
            logger.log("WorldModel/dino_reconstruction_loss", dino_reconstruction_loss.item())
            logger.log("WorldModelNorm/dino_norm", dino_norm.item())
      
        

    def update_wm(self, obs, action, reward, termination, logger=None, log_recs=False):
        # train wm
        self.eval()
        self.dist_head.train()
        self.storm_transformer.train()
        self.slots_head.train()
        self.reward_decoder.train()
        self.termination_decoder.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                embedding, _, _ = self.dino.dino_encode(obs)
            post_logits = self.dist_head.forward_post(embedding)
            flattened_sample = self.flatten_sample(self.stright_throught_gradient(post_logits, sample_mode="random_sample")) if self.conf.Models.WorldModel.stochastic_head else post_logits

            # transformer
            history_length = embedding.shape[1]
            src_length = tgt_length = history_length * self.conf.Models.Slot_attn.num_slots
            device = embedding.device
            positions = torch.arange(history_length - 1, -1, -1, device=device).repeat_interleave(self.conf.Models.Slot_attn.num_slots, dim=0).long() if self.conf.Models.WorldModel.slot_based  else torch.arange(src_length - 1, -1, -1, device=device) 
            temporal_mask = get_causal_mask(src_length, tgt_length, embedding.device, termination, self.conf.Models.Slot_attn.num_slots)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask, positions)

            # decoding reward and termination with dist_feat
            slots_hat = self.slots_head(dist_feat)
            combined_dist_feat = rearrange(dist_feat, 'b t s e-> b t (s e)') if self.conf.Models.WorldModel.wm_oc_pool_layer != 'cls-transformer' else dist_feat
            combined_dist_feat = self.wm_oc_pool_layer(combined_dist_feat)
            reward_hat = self.reward_decoder(combined_dist_feat)
            termination_hat = self.termination_decoder(combined_dist_feat)

            # STORM env losses
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            slots_loss = F.mse_loss(embedding.detach(), slots_hat)
            # dyn-rep loss
            wm_loss = reward_loss + slots_loss + termination_loss 

            # if stochastin representations used:
            if self.conf.Models.WorldModel.stochastic_head:
                prior_logits = self.dist_head.forward_prior(dist_feat)
                dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].reshape(-1,  *post_logits.shape[2:]).detach(), prior_logits.reshape(*post_logits.shape)[:, :-1].reshape(-1,  *post_logits.shape[2:]))
                representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].reshape(-1,  *post_logits.shape[2:]), prior_logits.reshape(*post_logits.shape)[:, :-1].reshape(-1,  *post_logits.shape[2:]).detach())
                wm_loss += 0.5*dynamics_loss + 0.1*representation_loss

        self.scaler.scale(wm_loss).backward()
        self.scaler.unscale_(self.wm_optimizer)  # for clip grad
        #trial_norm = self.trial_clip_grad_norm_(self.wm_parameters, max_norm=10.0)
        wm_norm = torch.nn.utils.clip_grad_norm_(self.wm_parameters, max_norm=10.0)
        self.scaler.step(self.wm_optimizer)
        self.scaler.update()

        if self.wm_scheduler is not None:
            self.wm_scheduler.step()

        self.wm_optimizer.zero_grad(set_to_none=True)

        ###########################
        ###########################
        # train decoder
        self.eval()
        self.image_decoder.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            obs_downsampled = self.downsample(rearrange(obs, 'b t c h w -> (b t) c h w'))
            obs_downsampled = rearrange(obs_downsampled, '(b t) c h w -> b t c h w', b=batch_size)

            #with torch.no_grad():
            #    embedding, _, _ = self.dino.dino_encode(obs.reshape(-1, 4, *obs.shape[2:]))
            #embedding = embedding.reshape(batch_size, batch_length, *embedding.shape[2:])

            # decoding image
            obs_hat, colors, masks = self.image_decoder(rearrange(embedding, 'b t s e -> (b t) s e').detach())

            # decoder losses
            obs_hat = rearrange(obs_hat, '(b t) c h w -> b t c h w', b=batch_size)
            decoder_loss = torch.pow(obs_hat - obs_downsampled, 2).mean()
 
        # gradient descent
        self.scaler.scale(decoder_loss).backward()
        self.scaler.unscale_(self.dec_optimizer)  # for clip grad
        dec_norm = torch.nn.utils.clip_grad_norm_(self.dec_parameters, max_norm=1.0)
        self.scaler.step(self.dec_optimizer)
        self.scaler.update()

        if self.dec_scheduler is not None:
            self.dec_scheduler.step()

        self.dec_optimizer.zero_grad(set_to_none=True)

        total_loss = wm_loss + decoder_loss

        if logger is not None:

            logger.log("WorldModel/decoder_reconstruction_loss", decoder_loss.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/slots_loss", slots_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            if self.conf.Models.WorldModel.stochastic_head:
                logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
                logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
                logger.log("WorldModel/representation_loss", representation_loss.item())
                logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
            logger.log("WorldModelNorm/wm_norm", wm_norm.item())
            logger.log("WorldModelNorm/dec_norm", dec_norm.item())
          
            if log_recs:
                #logger.log("WorldModel/predict_slots_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())
                obs_hat_list = self.compute_image_with_slots(obs_downsampled, obs_hat, rearrange(colors, '(b t) s c h w -> b t s c h w', b=batch_size), rearrange(masks, '(b t) s c h w -> b t s c h w', b=batch_size))
                logger.log("DINO/images/predict_slots_recs", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).squeeze(1).cpu().float().detach().numpy(), obs_hat.shape[1])



    def trial_clip_grad_norm_(self, 
            parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
            error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> torch.Tensor:
        r"""Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return torch.tensor(0.)
        first_device = grads[0].device
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
            = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]

        if norm_type == inf:
            norms = [g.detach().abs().max().to(first_device) for g in grads]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            norms = []
            for ((device, _), [grads]) in grouped_grads.items():
                if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                    norms.extend(torch._foreach_norm(grads, norm_type))
                elif foreach:
                    raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
                else:
                    norms.extend([torch.norm(g, norm_type) for g in grads])

            total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
            print(torch.stack([norm.to(first_device) for norm in norms]))

        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for ((device, _), [grads]) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                torch._foreach_mul_(grads, clip_coef_clamped.to(device))  # type: ignore[call-overload]
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in grads:
                    g.detach().mul_(clip_coef_clamped_device)

        return total_norm

    @torch.no_grad()
    def compute_image_with_slots(self, obs, recons, colors, masks):
        b, t, _, h, w = obs.size()
        plots = []
        obs, colors, masks = obs.cpu(), colors.cpu(), masks.cpu()
        recons = recons.cpu() if torch.is_tensor(recons) else recons
        for i in range(1):
            ob = obs[i] # (t c h w)
            recon = recons[i] # (t c h w)
            full_plot = torch.cat([ob.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
            color = colors[i].float() 
            mask = masks[i].float()
            subimage = color * mask
            mask = mask.repeat(1,1,3,1,1)
            full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
            full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
            full_plot = full_plot.view(-1, 1, 3, h, w)  # (H*W, 3, D, D)
            plots.append(full_plot)

        return plots