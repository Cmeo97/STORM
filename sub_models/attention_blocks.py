import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math
from functools import lru_cache
from PIL import Image

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    batch_size, batch_length = seq.shape[:2]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, batch_length, batch_length), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def get_subsequent_mask_with_batch_length(batch_length, device):
    ''' For masking out the subsequent info. '''
    subsequent_mask = (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=device), diagonal=1)).bool()
    return subsequent_mask


def get_causal_mask(src_length, tgt_length, device, stop_mask, num_current_tokens=0, mem_num_tokens=0, generation=False, slot_based=True):



    def save_grid_as_png(grid, filename='output.png'): # to test and visualize causal masks
        """
        Saves an NxN grid of booleans as a black and white PNG image.

        Parameters:
        - grid (list of lists or numpy array): The NxN grid of boolean values.
        - filename (str): The name of the file where the image will be saved.
        """
        # Convert the boolean grid to a numpy array if it isn't one already
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)

        # Normalize the boolean values to 0 and 255 (uint8 format)
        image_data = np.where(grid, 255, 0).astype(np.uint8)

        # Create an image from the array
        image = Image.fromarray(image_data, 'L')  # 'L' mode for grayscale

        # Save the image
        image.save(filename)

    def _get_base_mask_generation(src_length, tgt_length, device, num_current_tokens, slot_based):
        src_mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=device)
        delta_lengths = src_length - tgt_length
        if slot_based:
            for tgt_index in range(tgt_length):
                complete_square = (num_current_tokens - tgt_index % num_current_tokens)% num_current_tokens if (tgt_index+1)%num_current_tokens==0 and tgt_index>0 else (num_current_tokens - tgt_index % num_current_tokens)
                src_index = delta_lengths + tgt_index + complete_square 
                src_mask[tgt_index, :src_index] = False  # rows are targets, columns are sources            for tgt_index in range(tgt_length):
        else:
            for tgt_index in range(tgt_length):
                src_index = delta_lengths + tgt_index 
                src_mask[tgt_index, :src_index + 1] = False  # rows are targets, columns are sources
        return src_mask

    src_mask = _get_base_mask_generation(src_length, tgt_length, device, num_current_tokens, slot_based) 

    batch_size, seq_length = stop_mask.shape
    stop_mask = stop_mask.t()
    stop_mask_shift_right = torch.cat([stop_mask.new_zeros(1, batch_size), stop_mask], dim=0)
    stop_mask_shift_left = torch.cat([stop_mask, stop_mask.new_zeros(1, batch_size)], dim=0)

    tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril()
    src = torch.logical_and(stop_mask_shift_left.unsqueeze(0), tril.unsqueeze(-1))
    src = torch.cummax(src.flip(1), dim=1).values.flip(1)

    shifted_tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril(diagonal=-1)
    tgt = torch.logical_and(stop_mask_shift_right.unsqueeze(1), shifted_tril.unsqueeze(-1))
    tgt = torch.cummax(tgt, dim=0).values

    idx = torch.logical_and(tgt, src)[:-1, :-1] # remove extra dimensions 
    
    i, j, k = idx.shape 
    if slot_based:
        if generation:
            idx = idx.reshape(i, 1, j, 1, k).expand(i, num_current_tokens, j, (num_current_tokens + mem_num_tokens), k) \
                .reshape(i * num_current_tokens, j * (num_current_tokens + mem_num_tokens), k)
        else:
            idx = idx.reshape(i, 1, j, 1, k).expand(i, num_current_tokens, j, (num_current_tokens + min(mem_num_tokens, 1)), k) \
                .reshape(i * num_current_tokens, j * (num_current_tokens + min(mem_num_tokens, 1)), k)
    #else:
    #    if generation:
    #        idx = idx.reshape(i, 1, j, 1, k).expand(i, num_current_tokens, j, (num_current_tokens + mem_num_tokens), k) \
    #            .reshape(i * num_current_tokens, j * (num_current_tokens + mem_num_tokens), k)
    #    else:
    #        idx = idx.reshape(i, 1, j, 1, k).expand(i, num_current_tokens, j, (num_current_tokens + min(mem_num_tokens, 1)), k) \
    #            .reshape(i * num_current_tokens, j * (num_current_tokens + min(mem_num_tokens, 1)), k)

        
    
    idx = idx[-tgt_length:, -src_length:]

    src_mask = src_mask.unsqueeze(-1).tile(1, 1, batch_size)
    src_mask[idx] = True
    del stop_mask_shift_left, stop_mask_shift_right, tril, tgt, idx, src, shifted_tril
    #save_grid_as_png(src_mask[:,:,0].detach().cpu())
    if not slot_based:
        src_mask = src_mask.permute(2,0,1)

    return src_mask



def save_grid_as_png(grid, filename='output.png'): # to test and visualize causal masks
    """
    Saves an NxN grid of booleans as a black and white PNG image.
    
    Parameters:
    - grid (list of lists or numpy array): The NxN grid of boolean values.
    - filename (str): The name of the file where the image will be saved.
    """
    # Convert the boolean grid to a numpy array if it isn't one already
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    
    # Normalize the boolean values to 0 and 255 (uint8 format)
    image_data = np.where(grid, 255, 0).astype(np.uint8)
    
    # Create an image from the array
    image = Image.fromarray(image_data, 'L')  # 'L' mode for grayscale
    
    # Save the image
    image.save(filename)




def get_vector_mask(batch_length, device):
    mask = torch.ones((1, 1, batch_length), device=device).bool()
    # mask = torch.ones((1, batch_length, 1), device=device).bool()
    return mask


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            if attn.dtype == torch.float16:
                attn = attn.masked_fill(mask == 0, -6e4)
            else:
                attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(num_heads, feat_dim, feat_dim//num_heads, feat_dim//num_heads, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class AttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(num_heads, feat_dim, feat_dim//num_heads, feat_dim//num_heads, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask=None):
        output, attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn


class PositionalEncoding1D(nn.Module):
    def __init__(
        self,
        max_length: int,
        embed_dim: int
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def forward(self, feat):
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, :feat.shape[1], :]
        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, position:position+1, :]
        return feat


class RelativeMultiheadSelfAttention(nn.Module):

    def __init__(self, dim, head_dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1 / (dim ** 0.5)

        self.qkv_proj = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.pos_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        tgt_length, batch_size = x.shape[:2]
        pos_len = pos_encodings.shape[0]

        if mems is not None:
            cat = torch.cat([mems, x], dim=0)
            qkv = self.qkv_proj(cat)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q[-tgt_length:]
        else:
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        pos_encodings = self.pos_proj(pos_encodings)

        src_length = k.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = q.view(tgt_length, batch_size, num_heads, head_dim)
        k = k.view(src_length, batch_size, num_heads, head_dim)
        v = v.view(src_length, batch_size, num_heads, head_dim)
        pos_encodings = pos_encodings.view(pos_len, num_heads, head_dim)

        content_score = torch.einsum('ibnd,jbnd->ijbn', (q + u_bias, k))
        pos_score = torch.einsum('ibnd,jnd->ijbn', (q + v_bias, pos_encodings))
        pos_score = self._rel_shift(pos_score)

        # [tgt_length x src_length x batch_size x num_heads]
        attn_score = content_score + pos_score
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_score = attn_score.masked_fill(attn_mask[:, :, None, None], -float('inf'))
            elif attn_mask.ndim == 3:
                attn_score = attn_score.masked_fill(attn_mask[:, :, :, None], -float('inf'))

        # [tgt_length x src_length x batch_size x num_heads]
        attn = F.softmax(attn_score, dim=1)
        attn = self.dropout(attn)

        context = torch.einsum('ijbn,jbnd->ibnd', (attn, v))
        context = context.reshape(context.shape[0], context.shape[1], num_heads * head_dim)
        return self.dropout(self.out_proj(context))


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_length, dropout_p=0, batch_first=False):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        encodings = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encodings', encodings)

    def forward(self, positions):
        out = self.encodings[positions]
        out = self.dropout(out)
        return out.unsqueeze(0) if self.batch_first else out.unsqueeze(1)
