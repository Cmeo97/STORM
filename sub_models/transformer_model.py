import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import copy
from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache, RelativeMultiheadSelfAttention, PositionalEncoding


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask):
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

    def forward(self, samples, action, mask):
        '''
        Normal forward pass
        '''
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype, device):
        '''
        Reset self.kv_cache_list
        '''
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+1, samples.device)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats



class TransformerXL(nn.Module):

    def __init__(self, stoch_dim, action_dim, feat_dim, transformer_layer_config, num_layers, max_length, mem_length, batch_first=True, slot_based=True):
        super().__init__()
        
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        transformer_layer = TransformerXLDecoderLayer(**transformer_layer_config)
        self.layers = nn.ModuleList([copy.deepcopy(transformer_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.mem_length = mem_length
        self.batch_first = batch_first
        self.slot_based = slot_based
        self.pos_enc = PositionalEncoding(transformer_layer.embed_dim, max_length, dropout_p=transformer_layer.dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(transformer_layer.num_heads, transformer_layer.head_dim))
        self.v_bias = nn.Parameter(torch.Tensor(transformer_layer.num_heads, transformer_layer.head_dim))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def init_mems(self):
        if self.mem_length > 0:
            param = next(self.parameters())
            dtype, device = param.dtype, param.device
            mems = []
            for i in range(self.num_layers + 1):
                mems.append(torch.empty(0, dtype=dtype, device=device))
            return mems
        else:
            return None

    def forward(self, x, positions, attn_mask, mems=None, tgt_length=None, generation=False):
        if self.batch_first:
            x = x.transpose(0, 1)

        if mems is None:
            mems = self.init_mems()

        if tgt_length is None:
            tgt_length = x.shape[0]
        assert tgt_length > 0
      
        pos_enc = self.pos_enc(positions)
        hiddens = [x]
        #attentions = []
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out, pos_enc, self.u_bias, self.v_bias, attn_mask=attn_mask, mems=mems[i])
            hiddens.append(out)
            #attentions.append(attention)

        out = out[-tgt_length:]   #check in the tmw repository dimensionality of out!!

        if self.batch_first:
            out = out.transpose(0, 1)

        assert len(hiddens) == len(mems)
        if generation and self.slot_based:
            with torch.no_grad():
                for i in range(len(hiddens)):
                    mems[i] = torch.cat([mems[i], hiddens[i]], dim=0)[-self.mem_length:] 
        else:
            with torch.no_grad():
                for i in range(len(hiddens)):
                    mems[i] = torch.cat([mems[i], hiddens[i][0].unsqueeze(0)], dim=0)[-self.mem_length:] 
        return out, mems


def get_activation(nonlinearity, param=None):
    if nonlinearity is None or nonlinearity == 'none' or nonlinearity == 'linear':
        return nn.Identity()
    elif nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'leaky_relu':
        if param is None:
            param = 1e-2
        return nn.LeakyReLU(negative_slope=param)
    elif nonlinearity == 'elu':
        if param is None:
            param = 1.0
        return nn.ELU(alpha=param)
    elif nonlinearity == 'silu':
        return nn.SiLU()
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')
    

class TransformerXLDecoderLayer(nn.Module):

    def __init__(self, embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p, layer_norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.self_attn = RelativeMultiheadSelfAttention(embed_dim, head_dim, num_heads, dropout_p)
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        x = self.norm1(x + self.self_attn(x, pos_encodings, u_bias, v_bias, attn_mask, mems))
        x = self.norm2(x + self._ff(x))
        return x