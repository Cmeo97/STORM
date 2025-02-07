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

import agents
from math import sqrt
from sub_models.dino_sam import DinoSAM_OCextractor, SpatialBroadcastDecoder

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

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
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

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = self.unimix(logits)
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
    def __init__(self, in_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, conf):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        dtype = conf.BasicSettings.dtype
        self.use_amp = True if (dtype == 'torch.float16' or dtype == 'torch.bfloat16') else False
        self.tensor_dtype = torch.float16 if dtype == 'torch.float16' else torch.bfloat16 if dtype == 'torch.bfloat16' else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.conf = conf
        self.num_slots = self.conf.Models.Slot_attn.num_slots

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
                mem_length=conf.Models.WorldModel.wm_memory_length,
                conf=conf
            )
            self.image_decoder = SpatialBroadcastDecoder(
                resolution=conf.Models.Decoder.resolution, 
                dec_input_dim=conf.Models.Decoder.dec_input_dim,
                dec_hidden_dim=conf.Models.Decoder.dec_hidden_dim,
                out_ch=conf.Models.Decoder.out_ch
            )
            self.dist_head = DistHead(
                image_feat_dim=conf.Models.Slot_attn.token_dim,
                transformer_hidden_dim=transformer_hidden_dim,
                stoch_dim=self.stoch_dim
            )

            self.OC_pool_layer = nn.Sequential(
            nn.Linear(self.conf.Models.Slot_attn.num_slots*transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU()
            )

            self.slots_head = SlotsHead(
                embedding_size=conf.Models.Slot_attn.token_dim,
                transformer_hidden_dim=transformer_hidden_dim
            )

            self.downsample = Resize(size=(conf.Models.Decoder.resolution, conf.Models.Decoder.resolution))
     

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


        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode_obs(self, obs):
        if self.conf.Models.WorldModel.model=='OC-irisXL':
            with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
                slots, inits, z_vit = self.dino_encode(obs) 
                post_logits = self.dist_head.forward_post(slots)
                sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
                return slots, sample, inits, z_vit
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
                prior_logits = self.dist_head.forward_prior(dist_feat[:, -1:], generation=True)

                prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
                prior_flattened_sample = self.flatten_sample(prior_sample)
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
            prior_logits = self.dist_head.forward_prior(dist_feat, generation=True)

            # decoding
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                slots_hat = self.slots_head(dist_feat[:8]).permute(0,2,3,1)
                output_hat = self.image_decoder(slots_hat)
            else:
                output_hat = None
            
            if self.conf.Models.WorldModel.model=='OC-irisXL':
                combined_dist_feat = self.OC_pool_layer(rearrange(dist_feat, 'b t s e-> b t (s e)'))
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

    #@torch.no_grad()
    #def inspect_world_model_predictions(self, obs, obs_hat, colors=None, masks=None):
    #    
    #    b, t, _, h, w = obs.size()
    #    plots = []
    #    for i in range(b):
    #        obs = obs[i].cpu() # (t c h w)
    #        recon = obs_hat[i].cpu() # (t c h w)
    #        full_plot = torch.cat([obs.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
    #        color = colors[i].cpu()
    #        mask = masks[i].repeat(1,1,3,1,1).cpu()
    #        subimage = color * mask
    #        full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
    #        full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
    #        full_plot = full_plot.view(-1, 1, 3, h, w)  # (H*W, 3, D, D)
    #        plots.append(full_plot)
    #    
    #    return plots
    
    def compute_image_with_slots(self, obs, recons, colors=None, masks=None):
        b, t, _, h, w = obs.size()
        plots = []
   
        obs, recons, colors, masks = obs.cpu(), recons.cpu(), colors.cpu(), masks.cpu()

        for i in range(int(b/4 + 1)):
            ob = obs[i] # (t c h w)
            recon = recons[i] # (t c h w)
            full_plot = torch.cat([ob.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
            color = colors[i]
            mask = masks[i]
            subimage = color * mask
            mask = mask.repeat(1,1,3,1,1)
            full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
            full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
            full_plot = full_plot.view(-1, 1, 3, h, w)  # (H*W, 3, D, D)
            plots.append(full_plot)

        return plots

       
        

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
                latent_size = (imagine_batch_size, imagine_batch_length+1, self.num_slots, self.stoch_flattened_dim)
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
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat, mems = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], self.termination_hat_buffer[:, i:i+1] ,log_video=log_video, mems=mems, device=device)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                last_obs_hat, last_colors_hat, last_masks_hat = last_obs_hat
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16].unsqueeze(1))  # uniform sample vec_env
                colors_hat_list.append(last_colors_hat[::imagine_batch_size//16].unsqueeze(1)) 
                masks_hat_list.append(last_masks_hat[::imagine_batch_size//16].unsqueeze(1)) 

        
        if log_video:
            obs_hat = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1)
            obs_downsampled = self.downsample(rearrange(obs_gt[:obs_hat_list[0].shape[0]], 'b t c h w->(b t c) h w')).view(*obs_hat.shape)
            colors_hat, masks_hat = torch.cat(colors_hat_list, dim=1), torch.cat(masks_hat_list, dim=1)
            logger.log("Imagine/predict_video", obs_hat.cpu().float().detach().numpy())
            obs_hat_list = self.compute_image_with_slots(obs_downsampled, obs_hat, colors_hat, masks_hat)
            logger.log("Imagine/predict_slots_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def update(self, obs, action, reward, termination, logger=None, log_recs=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=self.tensor_dtype, enabled=self.use_amp):
            # encoding
            if self.conf.Models.WorldModel.model == 'OC-irisXL':
                embedding, _, z_vit = self.dino_encode(obs)  # embedding = slots
                reconstructions = self.dino.decode(embedding)
                history_length = embedding.shape[1]
                src_length = tgt_length = history_length * self.conf.Models.Slot_attn.num_slots
                device = embedding.device
                positions = torch.arange(history_length - 1, -1, -1, device=device).repeat_interleave(self.conf.Models.Slot_attn.num_slots, dim=0).long() if self.conf.Models.WorldModel.slot_based  else torch.arange(src_length - 1, -1, -1, device=device) 
            else:
                embedding = self.encoder(obs)
                
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            
            flattened_sample = self.flatten_sample(sample) if self.conf.Models.WorldModel.use_onehot else embedding.detach()
            flattened_sample = flattened_sample.detach() if self.conf.Models.WorldModel.independent_modules else flattened_sample
            # decoding image
            obs_hat, colors, masks = self.image_decoder(rearrange(embedding, 'B T S E-> (B T) S E').unsqueeze(-1).detach()) if self.conf.Models.WorldModel.model=='OC-irisXL' else self.image_decoder(flattened_sample)
            
            # transformer
            temporal_mask = get_causal_mask(src_length, tgt_length, embedding.device, termination, self.conf.Models.Slot_attn.num_slots) if self.conf.Models.WorldModel.model == 'OC-irisXL' else get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask, positions) if self.conf.Models.WorldModel.model == 'OC-irisXL' else self.storm_transformer(flattened_sample, action, temporal_mask) 
            prior_logits = self.dist_head.forward_prior(dist_feat)
            # decoding reward and termination with dist_feat
            if self.conf.Models.WorldModel.model=='OC-irisXL':
                slots_hat = self.slots_head(dist_feat)
                combined_dist_feat = self.OC_pool_layer(rearrange(dist_feat, 'b t s e-> b t (s e)'))
                reward_hat = self.reward_decoder(combined_dist_feat)
                termination_hat = self.termination_decoder(combined_dist_feat)
            else:
                reward_hat = self.reward_decoder(dist_feat)
                termination_hat = self.termination_decoder(dist_feat)

            # DINO losses
            consistency_loss = self.dino.cosine_loss(embedding) if self.conf.Models.WorldModel.model == 'OC-irisXL' else 0
            dino_reconstruction_loss = torch.pow(z_vit - reconstructions, 2).mean() if self.conf.Models.WorldModel.model == 'OC-irisXL' else 0
            # STORM env losses
            decoder_loss = self.mse_loss_func(rearrange(obs_hat, 'b c h w->(b c) h w').unsqueeze(0).unsqueeze(0), self.downsample(rearrange(obs, 'b t c h w->(b t c) h w').mul(2).sub(1)).unsqueeze(0).unsqueeze(0))
            dino_loss = dino_reconstruction_loss + consistency_loss if self.conf.Models.WorldModel.model == 'OC-irisXL' else self.mse_loss_func(obs_hat, obs)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            slots_loss = F.mse_loss(embedding.detach(), slots_hat)
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].reshape(-1,  *post_logits.shape[2:]).detach(), prior_logits.reshape(*post_logits.shape)[:, :-1].reshape(-1,  *post_logits.shape[2:]))
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].reshape(-1,  *post_logits.shape[2:]), prior_logits.reshape(*post_logits.shape)[:, :-1].reshape(-1,  *post_logits.shape[2:]).detach())
            total_loss = dino_loss + decoder_loss + reward_loss + slots_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

        # gradient descent
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
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/slots_loss", slots_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            logger.log("WorldModel/representation_loss", representation_loss.item())
            logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
            if log_recs:
                #logger.log("WorldModel/predict_slots_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())
                obs_hat = obs_hat.view(obs.shape[0], -1, *obs_hat.shape[1:])
                obs_downsampled = self.downsample(rearrange(obs, 'b t c h w->(b t c) h w').mul(2).sub(1)).view(*obs_hat.shape)
                obs_hat_list = self.compute_image_with_slots(obs_downsampled, obs_hat, rearrange(colors, '(b t) s c h w->b t s c h w', b=obs.shape[0]) , rearrange(masks, '(b t) s c h w->b t s c h w', b=obs.shape[0]))
                logger.log("DINO/video/predict_slots_recs", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())

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
        
        if self.dino.run_in_eval_mode and self.dino.training:
            self.dino.eval()
  
        if self.dino.vit_freeze:
            # Speed things up a bit by not requiring grad computation.
            with torch.no_grad():
                features = self.dino.vit.forward_features(x)
        else:
            features = self.dino.vit.forward_features(x)
  
        if self.dino._feature_hooks is not None:
            hook_features = [hook.pop() for hook in self.dino._feature_hooks]
  
        if len(self.dino.feature_levels) == 0:
            # Remove class token when not using hooks.
            features = features[:, 1:]
            positions = _transformer_compute_positions(features)
        else:
            features = hook_features[: len(self.dino.feature_levels)]
            positions = _transformer_compute_positions(features[0])
            features = torch.cat(features, dim=-1)
  
        return features, positions
  
    def dino_encode(self, x: torch.Tensor):
    
        shape = x.shape  # (..., C, H, W)
        x = x.reshape(-1, *shape[-3:])
        z, _ = self.vit_encode(x)
        z = self.dino.pos_embed(z)
  
        if self.dino.slot_attn.is_video:
            z = z.reshape(*shape[:-3], *z.shape[1:]) # video
            if z.dim() == 3:
                z = z.unsqueeze(1)
        z, inits = self.dino.slot_attn(z)
        if self.dino.slot_attn.is_video:
            z = z.view(-1, *z.shape[-2:]) # video
            inits = inits.view(-1, *inits.shape[-2:]) # video
  
        z_vit, _ = self.vit_encode(x)
        
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        inits = inits.reshape(*shape[:-3], *inits.shape[1:])
        z_vit = z_vit.reshape(*shape[:-3], *z_vit.shape[1:])
  
        return z, inits, z_vit
    
    @torch.no_grad()
    def compute_image_with_slots(self, obs, recons, colors, masks):
        b, t, _, h, w = obs.size()
        plots = []
        obs, colors, masks = obs.cpu(), colors.cpu(), masks.cpu()
        recons = recons.cpu() if torch.is_tensor(recons) else recons
        for i in range(int(b/4 + 1)):
            ob = obs[i] # (t c h w)
            recon = recons[i] # (t c h w)
            full_plot = torch.cat([ob.unsqueeze(1), recon.unsqueeze(1)], dim=1) # (t 2 c h w)
            color = colors[i]
            mask = masks[i]
            subimage = color * mask
            mask = mask.repeat(1,1,3,1,1)
            full_plot = torch.cat([full_plot, mask, subimage], dim=1) #(T,2+K+K,3,D,D)
            full_plot = full_plot.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
            full_plot = full_plot.view(-1, 1, 3, h, w)  # (H*W, 3, D, D)
            plots.append(full_plot)

        return plots