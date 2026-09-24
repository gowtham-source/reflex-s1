"""Shared-state, decision-level MoR with sparse top-2 MoE; no text generation."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig

@dataclass
class ModelConfig:
    width: int = 128
    experts: int = 4
    top_k: int = 2
    depth: int = 3
    heads: int = 4
    max_length: int = 256
    condition_question: bool = False
    pair_segments: bool = False

    def __post_init__(self):
        if self.depth < 1 or not 1 <= self.top_k <= self.experts or self.width % self.heads or self.max_length < 2:
            raise ValueError("Invalid model configuration")

class SparseExperts(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.router = nn.Linear(d, n)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d)) for _ in range(n)])
        self.k = k
    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat).float()
        probabilities = router_logits.softmax(-1)
        weights, indices = probabilities.topk(self.k, -1)
        weights = weights / weights.sum(-1, keepdim=True)
        result = torch.zeros_like(flat)
        # Dispatch only selected tokens; no expert capacity drops.
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == i)
            if rows.numel():
                values = expert(flat[rows]) * weights[rows, slots, None].to(flat.dtype)
                result.index_add_(0, rows, values.to(result.dtype))
        frequency = F.one_hot(indices, len(self.experts)).float().mean((0,1))
        balance = len(self.experts) * (probabilities.mean(0) * frequency.detach()).sum()
        zloss = router_logits.logsumexp(-1).square().mean()
        return result.reshape(shape), balance + 0.001*zloss

class RecursiveBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True, dropout=0)
        self.moe = SparseExperts(d, cfg.experts, cfg.top_k)
    def forward(self, q, state, mask):
        z, _ = self.attn(self.norm1(q), state, state, key_padding_mask=~mask.bool(), need_weights=False)
        q = q + z
        z, aux = self.moe(self.norm2(q))
        return q + z, aux

class DecisionModel(nn.Module):
    def __init__(self, encoder_path, cfg=None, pretrained=True):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = (AutoModel.from_pretrained(encoder_path, attn_implementation='sdpa') if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(encoder_path), attn_implementation='sdpa'))
        if self.cfg.max_length > self.encoder.config.max_position_embeddings:
            raise ValueError('Requested context exceeds encoder position capacity')
        h, d = self.encoder.config.hidden_size, self.cfg.width
        self.projection = nn.Linear(h, d)
        self.state_projection = nn.Linear(h, d)
        self.block = RecursiveBlock(self.cfg)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.depth_policy = nn.Sequential(nn.Linear(h*2, d), nn.Tanh(), nn.Linear(d, self.cfg.depth))
        self.semantic_scale = nn.Parameter(torch.tensor(10.0))
        self.type_emb = nn.Embedding(3, d)

    def encode(self, tokens):
        h = self.encoder(**tokens).last_hidden_state
        mask = tokens['attention_mask']
        pooled = (h * mask[...,None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return h, pooled

    def features(self, states, candidates, type_id=0, option_features=None, state_features=None):
        h, pooled = self.encode(states) if state_features is None else state_features
        _, options = self.encode(candidates) if option_features is None else (None, option_features)
        if options.ndim == 2:
            options = options.unsqueeze(0).expand(h.shape[0], -1, -1)
        policy = self.depth_policy(torch.cat([pooled, options.mean(1)], -1))
        q = self.projection(options) + self.state_projection(pooled)[:,None,:]
        q = q + self.type_emb.weight[type_id]
        semantic = F.cosine_similarity(pooled[:,None,:], options, dim=-1) * self.semantic_scale.clamp(1,30)
        return q, self.projection(h), semantic, policy

    def forward(self, states, candidates, type_id=0, option_features=None, adaptive=False, force_depth=None, state_features=None):
        q, h, base, policy = self.features(states, candidates, type_id, option_features, state_features)
        if adaptive:
            depths = policy.argmax(-1) + 1 if force_depth is None else torch.full((q.shape[0],), force_depth, device=q.device)
            result = base.clone()
            for step in range(self.cfg.depth):
                active = (depths > step).nonzero().flatten()
                if not active.numel():
                    break
                next_q, _ = self.block(q[active], h[active], states['attention_mask'][active])
                q = q.index_copy(0, active, next_q)
                result = result.index_copy(0, active, base[active] + self.scorer(next_q).squeeze(-1).float())
            return {'logits': result, 'depths': depths, 'policy': policy}
        logits, aux = [], 0
        for _ in range(self.cfg.depth):
            q, balance = self.block(q, h, states['attention_mask'])
            logits.append(base + self.scorer(q).squeeze(-1).float())
            aux = aux + balance
        return {'all_logits': torch.stack(logits, 1), 'policy': policy, 'aux': aux/self.cfg.depth}

    def save(self, directory, tokenizer):
        from pathlib import Path
        import json
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(p/'encoder')
        tokenizer.save_pretrained(p/'encoder')
        (p/'config.json').write_text(json.dumps(asdict(self.cfg), indent=2))
        torch.save(self.state_dict(), p/'model.pt')
        (p/'calibration.json').unlink(missing_ok=True)

    @classmethod
    def load(cls, directory, device='cpu'):
        from pathlib import Path
        import json
        p=Path(directory)
        model=cls(p/'encoder', ModelConfig(**json.loads((p/'config.json').read_text())), pretrained=False)
        weights_path = p/'model.safetensors' if (p/'model.safetensors').exists() else p/'model.pt'
        if str(weights_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(str(weights_path), device='cpu'))
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        return model.to(device).eval()