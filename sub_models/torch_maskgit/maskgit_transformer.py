import math
import torch
import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)


class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(self, dim=768, heads=8, attention_dropout=0.1):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        att = q @ torch.transpose(k, 1, 2) / self.norm
        qk = torch.softmax(att, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(self, dim=768, heads=8, attention_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, heads, attention_dropout) for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class RightShift(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(RightShift, self).__init__()
        self.dtype = dtype
        self.sos = None

    def forward(self, x):
        if self.sos is None:
            self.sos = nn.Parameter(torch.randn(x.shape[-1], dtype=self.dtype) * 0.02)
        
        sos = self.sos.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([sos, x[:, :-1]], dim=1)
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_dim, dropout, attention_dropout, 
                 vocab_size=None, vocab_dim=None, input_dim=None, shape=None, pos_embed_type='absolute',
                 out_dim=None, use_fc_in=True, right_shift=False, dtype=torch.float32):
        super(Transformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.input_dim = input_dim
        self.use_fc_in = use_fc_in
        self.right_shift = right_shift
        self.pos_embed_type = pos_embed_type
        self.out_dim = out_dim
        self.shape = shape
        self.dtype = dtype
        
        if self.vocab_size and self.vocab_dim:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vocab_dim)
            
        if self.use_fc_in:
            fc_in_dim = self.vocab_dim

            if not hasattr(self, 'embedding'):
                fc_in_dim += self.input_dim
            self.fc_in = nn.Linear(fc_in_dim, self.embed_dim)
      
        self.broadcast_position_biases = BroadcastPositionBiases(shape=self.shape,
                                                                 embed_dim=self.embed_dim,
                                                    dtype=self.dtype)
        
        self.rshift = RightShift(dtype=self.dtype)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, mlp_dim, dropout, attention_dropout) 
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        if self.out_dim:
            self.fc_final = nn.Linear(self.embed_dim, self.out_dim, bias=False)
            self.bias_adder = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, inputs, cond=None):
        x = inputs
        
        if hasattr(self, 'embedding'):
            x = self.embedding(x)

        if cond is not None:
            x = torch.cat((x, cond), dim=-1) 

        if hasattr(self, 'fc_in'):
            x = self.fc_in(x)

        old_shape = x.shape[1:-1]
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.right_shift:
            x = self.rshift(x)

        position_bias = self.broadcast_position_biases(x)
        x = x + position_bias
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.fc_out(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if hasattr(self, 'fc_final'):
            x = self.fc_final(x) + self.bias_adder
        
        x = x.reshape(x.shape[0], *old_shape, x.shape[-1])
        return x 

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, attention_dropout):
        super(TransformerLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
       
        self.mha = MultiHeadAttention(self.embed_dim, self.num_heads, self.attention_dropout)

        self.dropout_x = nn.Dropout(p=self.dropout)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

  
        self.mlp = MlpBlock(self.embed_dim, self.dropout)
        self.dropout_y = nn.Dropout(p=self.dropout)

    def forward(self, inputs):
        x = self.layer_norm1(inputs)
        x = self.mha(x)
        x = self.dropout_x(x) + inputs

        y = self.layer_norm2(x)
        y = self.mlp(y)
        y = self.dropout_y(y)

        return y



class MlpBlock(nn.Module):
    def __init__(self, intermediate_dim, dropout_rate=0.1, dtype=torch.float32):
        super(MlpBlock, self).__init__()
        self.dtype = dtype
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_input = nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        self.linear_output = nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        
    def forward(self, inputs):
        x = self.linear_input(inputs)
        x = self.gelu(x) 
        x = self.dropout(x) 
        x = self.linear_output(x)
       
        return x


class BroadcastPositionBiases(nn.Module):
    def __init__(self, embed_dim, shape=None, dtype=torch.float32):
        super(BroadcastPositionBiases, self).__init__()
        self.shape = shape
        self.dtype = dtype
        embs = [
            nn.Parameter(torch.randn(32, embed_dim, dtype=self.dtype) * 0.02)
            for i in range(1)
        ]
        self.embs = nn.ParameterList(embs)


    def forward(self, x):
        if self.shape is None:
            shape = x.shape[1:-1]
        else:
            shape = self.shape
        n_dim = len(shape)
        embed_dim = x.shape[-1]

        chunk_sizes = [embed_dim // n_dim + (i < (embed_dim % n_dim))
                       for i in range(n_dim)]
        assert sum(chunk_sizes) == embed_dim, f'sum({chunk_sizes}) = {sum(chunk_sizes)} != {embed_dim}'

        out = []
        for i in range(n_dim):
            e = self.embs[i]
            e = e.reshape((1,) + (1,) * i + (shape[i],) + (1,) * (n_dim - i - 1) + (-1,))
            e = e.expand(-1, *shape, e.shape[-1])
            out.append(e)


        out = torch.cat(out, dim=-1)
        out = out.contiguous().view(np.prod(shape), embed_dim)
        return out.to(x.device)