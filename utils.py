import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value, n_cols=8):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag], n_cols=n_cols)
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False
    conf.BasicSettings.dtype = 'bfloat16'
    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.Transformer = None
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0
    conf.Models.WorldModel.tokens_per_block = 8  #17
    conf.Models.WorldModel.max_blocks = 20
    conf.Models.WorldModel.attention = 'causal'
    conf.Models.WorldModel.num_layers = 10
    conf.Models.WorldModel.num_heads = 4
    conf.Models.WorldModel.embed_dim = 256
    conf.Models.WorldModel.embed_pdrop = 0.1
    conf.Models.WorldModel.resid_pdrop = 0.1
    conf.Models.WorldModel.attn_pdrop = 0.1
    conf.Models.WorldModel.model = 'OC-irisXL'
    conf.Models.WorldModel.continuos_embed_dim = 128
    conf.Models.WorldModel.dyn_num_heads = 4
    conf.Models.WorldModel.dyn_num_layers = 10
    conf.Models.WorldModel.dyn_feedforward_dim = 1024
    conf.Models.WorldModel.dyn_head_dim = 64
    conf.Models.WorldModel.dyn_z_dims = [512, 512, 512, 512]
    conf.Models.WorldModel.dyn_reward_dims = [256, 256, 256, 256]
    conf.Models.WorldModel.dyn_discount_dims = [256, 256, 256, 256]
    conf.Models.WorldModel.dyn_input_rewards = True
    conf.Models.WorldModel.dyn_input_discounts = False
    conf.Models.WorldModel.dyn_act = 'silu'
    conf.Models.WorldModel.dyn_norm = 'none'
    conf.Models.WorldModel.dyn_dropout = 0.1
    conf.Models.WorldModel.dyn_lr = 1e-4
    conf.Models.WorldModel.dyn_wd = 1e-6
    conf.Models.WorldModel.dyn_eps = 1e-5
    conf.Models.WorldModel.dyn_grad_clip = 100
    conf.Models.WorldModel.dyn_z_coef = 1
    conf.Models.WorldModel.dyn_reward_coef = 10
    conf.Models.WorldModel.dyn_discount_coef = 50
    conf.Models.WorldModel.wm_batch_size = 100
    conf.Models.WorldModel.wm_sequence_length = 340
    conf.Models.WorldModel.wm_train_steps = 1
    conf.Models.WorldModel.wm_memory_length = 8
    conf.Models.WorldModel.wm_discount_threshold = 0.1
    conf.Models.WorldModel.regularization_post_quant = False
    conf.Models.WorldModel.regularization_tokens = False
    conf.Models.WorldModel.regularization_embeddings = False
    conf.Models.WorldModel.embedding_input = False
    conf.Models.WorldModel.slot_based = False
    conf.Models.WorldModel.slot_regularization = False
    conf.Models.WorldModel.regularization_k_pred = False
    conf.Models.WorldModel.vit_model_name = 'samvit_base_patch16'
    conf.Models.WorldModel.vit_use_pretrained = True
    conf.Models.WorldModel.vit_freeze = True
    conf.Models.WorldModel.vit_feature_level = 12
    conf.Models.WorldModel.transformer_layer = CN()
    conf.Models.WorldModel.independent_modules = True
    conf.Models.WorldModel.stochastic_head = False
    conf.Models.WorldModel.stochastic_dim = 32
    conf.Models.WorldModel.action_emb_dim = 128
    conf.Models.WorldModel.wm_oc_pool_layer = 'cls-transformer'

    
    conf.Models.WorldModel.transformer_layer.embed_dim = 256
    conf.Models.WorldModel.transformer_layer.feedforward_dim = 1024
    conf.Models.WorldModel.transformer_layer.head_dim = 64
    conf.Models.WorldModel.transformer_layer.num_heads = 4
    conf.Models.WorldModel.transformer_layer.activation = 'silu' 
    conf.Models.WorldModel.transformer_layer.dropout_p = 0.1
    conf.Models.WorldModel.transformer_layer.layer_norm_eps = 1e-5

    conf.Models.Decoder = CN()
    conf.Models.Decoder.resolution = 224
    conf.Models.Decoder.dec_input_dim = 128
    conf.Models.Decoder.dec_hidden_dim = 64
    conf.Models.Decoder.out_ch = 4
    conf.Models.Decoder.vit_num_patches = 196 # res 224
    conf.Models.Decoder.dec_hidden_layers = [1024, 1024, 1024] # MLPDecoder
    conf.Models.Decoder.dec_output_dim = 768 # MLPDecoder

    conf.Models.Slot_attn = CN()
    conf.Models.Slot_attn.num_slots = 7 # num_tokens = num_slots * tokens_per_slot
    conf.Models.Slot_attn.tokens_per_slot = 1
    conf.Models.Slot_attn.iters = 3
    conf.Models.Slot_attn.channels_enc = 128
    conf.Models.Slot_attn.token_dim = 128 # need to match embed_dim if no pre_process_conv
    conf.Models.Slot_attn.prior_class = 'grucell'
    conf.Models.Slot_attn.pred_prior_from = 'last'

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0
    conf.Models.Agent.pooling_layer = 'dino-sbd'

    conf.Models.CLSTransformer = CN()
    conf.Models.CLSTransformer.NumLayers = 0
    conf.Models.CLSTransformer.HiddenDim = 256
    conf.Models.CLSTransformer.NumHeads = 4

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
