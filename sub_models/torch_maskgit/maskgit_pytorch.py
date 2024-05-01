import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel, Categorical
from .maskgit_transformer import Transformer 

MASK_ID = -1


def topk_sample(logits, top_k=None):
    if top_k is not None:
        top_k = min(top_k, logits.shape[-1])
        # Getting the kth highest value to use as a threshold
        indices_to_remove = logits < torch.topk(logits ,top_k)[..., -1, None]
        logits = torch.where(indices_to_remove, torch.finfo(logits.dtype).min, logits)
   
    return Categorical(logits=logits).sample()


def schedule(ratio, total_unknown, method='cosine'):
    if method == 'uniform':
        mask_ratio = 1. - ratio
    elif 'pow' in method:
        exponent = float(method.replace('pow', ''))
        mask_ratio = 1. - ratio ** exponent
    elif method == 'cosine':
        mask_ratio = torch.cos(math.pi / 2. * ratio)
    elif method == 'log':
        mask_ratio = -torch.log2(ratio) / torch.log2(total_unknown)
    elif method == 'exp':
        mask_ratio = 1 - torch.exp2(-torch.log2(total_unknown) * (1 - ratio))
    
    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.)

    return mask_ratio

    
def mask_by_random_topk(mask_len, probs, temperature=1.0):
    gumble = Gumbel(0, 1)
    confidence = torch.log(probs) + temperature * gumble.sample(probs.shape).to(probs.device)
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    cut_off = sorted_confidence[..., mask_len]
    masking = (confidence < cut_off)

    return masking


# fix input args
def sample_mask(Z, T, device):
    N = np.prod(Z)
    idxs = torch.arange(N)
    idxs = idxs[torch.randperm(N)]
    chunks = torch.chunk(idxs, T)

    masks = []

    for t in range(T):
        mask = F.one_hot(chunks[t], N).sum(dim=0).bool()
        mask = mask.view(Z).to(device)
        masks.append(mask)

    return masks


class MaskGit(nn.Module):
    def __init__(self, shape, vocab_size, vocab_dim, mask_schedule, tfm_kwargs, dtype=torch.float32):
        super(MaskGit, self).__init__()
        self.shape = shape
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.mask_schedule = mask_schedule
        self.tfm_kwargs = tfm_kwargs
        self.dtype = dtype

        self.token_embed = nn.Parameter(torch.randn([self.vocab_size + 1, self.vocab_dim], dtype=self.dtype) * 0.02)
        self.net = Transformer(**self.tfm_kwargs, shape=self.shape, pos_embed_type='broadcast', dtype=self.dtype)
        self.mlm = MlmLayer(self.tfm_kwargs['embed_dim'], self.vocab_dim, self.vocab_size, self.dtype)


    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / 32 + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def _step(self, x, cond=None):
        x = self.token_embed[x]
        x = self.net(x, cond=cond)

        # Only try to predict real tokens, not masks
        logits = self.mlm(x, self.token_embed[:self.vocab_size])
        logits = self.unimix(logits)

        return logits
        
    def sample(self, n, T_draft, T_revise, M, cond=None, sample_shape=None):
        device = self.token_embed.device
        sample = torch.full((n, sample_shape), MASK_ID, dtype=torch.int32).to(device)

        def _update(samples, masks):
            for mask in masks:
                samples = torch.where(mask, MASK_ID, samples)
                logits = self._step(samples, cond=cond)
                s = topk_sample(logits)  # Assuming you have a PyTorch version for this
                samples = torch.where(mask, s, samples)

            return samples
        
        # Draft
        masks = sample_mask(sample_shape, T_draft, device=device)
        sample = _update(sample, masks)
        
        # Revise
        for _ in range(M):
            masks = sample_mask(sample_shape, T_revise, device=device)
            sample = _update(sample, masks)

        return sample

    def forward(self, x, cond=None):
        B, L = x.shape[0], torch.prod(torch.tensor(x.shape[1:]))
        ratio = torch.rand((B,), dtype=self.dtype).to(x.device)
        ratio = schedule(ratio, L, method=self.mask_schedule)
        ratio = torch.maximum(torch.tensor(1).to(x.device), torch.floor(ratio * L))

        sample = torch.arange(L).repeat(B, 1).to(x.device)
        for b in range(B):
            sample[b] = sample[b,torch.randperm(L)]

        mask = sample < ratio[:, None]
        mask = mask.reshape(x.shape)

        masked_x = torch.where(mask, MASK_ID, x)
        #print(f"shape of masked_x: {masked_x.shape}, cond: {cond.shape}")
        logits = self._step(masked_x, cond=cond)
        labels = F.one_hot(x, num_classes=self.vocab_size)
        return logits, labels, mask

        
class MlmLayer(nn.Module):
    def __init__(self, embed_dim, vocab_dim, vocab_size, dtype=torch.float32):
        super(MlmLayer, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_dim = vocab_dim
        self.dtype = dtype
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embed_dim, vocab_dim)
        self.ln = nn.LayerNorm(vocab_dim)
        # Not sure whether AddBias should have shape vocab_size; I think it should be vocab_dim
        self.bias = AddBias(vocab_size, dtype)

    def forward(self, x, embeddings):
        x = self.gelu(self.fc(x))
        x = self.ln(x)
        logits = torch.matmul(x, embeddings.t())
        logits = self.bias(logits)
        return logits


class AddBias(nn.Module):
    def __init__(self, output_dim, dtype):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=dtype))
    
    def forward(self, x):
        return x + self.bias