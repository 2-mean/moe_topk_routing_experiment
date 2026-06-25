from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    n_experts: int
    expert_hidden: int
    dropout: float = 0.0


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, expert_hidden: int, dropout: float) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, expert_hidden, dropout) for _ in range(n_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k: int,
        collect_routes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        if top_k < 1 or top_k > self.n_experts:
            raise ValueError(f"top_k must be in [1, {self.n_experts}], got {top_k}")

        gate_logits = self.router(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_logits, top_ids = torch.topk(gate_logits, k=top_k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        gather_index = top_ids.unsqueeze(-1).expand(*top_ids.shape, x.shape[-1])
        selected_outputs = torch.gather(expert_outputs, dim=2, index=gather_index)
        mixed = (selected_outputs * top_weights.unsqueeze(-1)).sum(dim=2)

        mean_probs = gate_probs.mean(dim=(0, 1))
        aux_loss = ((mean_probs - (1.0 / self.n_experts)) ** 2).sum() * self.n_experts

        route = None
        if collect_routes:
            route = {
                "gate_logits": gate_logits.detach(),
                "gate_probs": gate_probs.detach(),
                "selected_ids": top_ids.detach(),
                "selected_weights": top_weights.detach(),
            }
        return mixed, aux_loss, route


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = TopKMoE(
            config.d_model,
            config.n_experts,
            config.expert_hidden,
            config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k: int,
        causal_mask: torch.Tensor,
        collect_routes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        moe_out, aux_loss, route = self.moe(self.ln2(x), top_k=top_k, collect_routes=collect_routes)
        x = x + moe_out
        return x, aux_loss, route


class TinyMoETransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        top_k: int,
        collect_routes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds configured {self.config.seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )

        aux_losses = []
        routes = []
        for layer_id, block in enumerate(self.blocks):
            x, aux_loss, route = block(
                x,
                top_k=top_k,
                causal_mask=causal_mask,
                collect_routes=collect_routes,
            )
            aux_losses.append(aux_loss)
            if route is not None:
                route["layer_id"] = torch.tensor(layer_id)
                routes.append(route)

        logits = self.lm_head(self.ln_f(x))
        aux = torch.stack(aux_losses).mean()
        return logits, aux, routes

