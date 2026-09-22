"""
COLLISION Modeling for Hugging Face Transformers Integration.
Enables AutoModelForCausalLM.from_pretrained("collision-10M/Collision-1B", trust_remote_code=True)
and pipeline("text-generation", model="collision-10M/Collision-1B", trust_remote_code=True).
"""

import math
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
    from transformers.generation import GenerationMixin
except ImportError:
    class PreTrainedModel(nn.Module):
        config_class = None
        base_model_prefix = "model"
        def __init__(self, config, *inputs, **kwargs):
            super().__init__()
            self.config = config

    class GenerationMixin:
        pass

    class CausalLMOutputWithPast:
        def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None):
            self.loss = loss
            self.logits = logits
            self.past_key_values = past_key_values
            self.hidden_states = hidden_states
            self.attentions = attentions

    class BaseModelOutputWithPast:
        def __init__(self, last_hidden_state=None, past_key_values=None, hidden_states=None, attentions=None):
            self.last_hidden_state = last_hidden_state
            self.past_key_values = past_key_values
            self.hidden_states = hidden_states
            self.attentions = attentions

try:
    from .configuration_collision import CollisionConfig
except (ImportError, ValueError):
    from configuration_collision import CollisionConfig


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.emb(positions)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
            persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.size()

        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, -1e4)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, T) -> (B, 1, 1, T)
                mask = attention_mask[:, None, None, :]
                scores = scores.masked_fill(mask == 0, -1e4)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)

        outputs = (out,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_outputs = self.attn(self.ln1(x), attention_mask=attention_mask, output_attentions=output_attentions)
        x = x + attn_outputs[0]
        x = x + self.ffn(self.ln2(x))
        outputs = (x,) + attn_outputs[1:]
        return outputs


class CollisionPreTrainedModel(PreTrainedModel):
    config_class = CollisionConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            emb_weight = module.emb.weight if hasattr(module, "emb") else module.weight
            torch.nn.init.normal_(emb_weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def post_init(self):
        self.apply(self._init_weights)


class CollisionModel(CollisionPreTrainedModel):
    def __init__(self, config: CollisionConfig):
        super().__init__(config)
        self.config = config
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                max_seq_len=config.max_seq_len,
                d_ff=config.d_ff,
                dropout=config.dropout
            ) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.token_emb.emb

    def set_input_embeddings(self, value):
        self.token_emb.emb = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, "output_attentions") else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, "output_hidden_states") else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, "use_return_dict") else True

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            x = self.token_emb(input_ids)
            if position_ids is None:
                x = x + self.pos_emb(input_ids)
            else:
                x = x + self.pos_emb.emb(position_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
            x = inputs_embeds
            if position_ids is None:
                positions = torch.arange(seq_length, device=device).unsqueeze(0)
                x = x + self.pos_emb.emb(positions)
            else:
                x = x + self.pos_emb.emb(position_ids)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        x = self.drop(x)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            layer_outputs = block(x, attention_mask=attention_mask, output_attentions=output_attentions)
            x = layer_outputs[0]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        x = self.ln_f(x)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        if not return_dict:
            return tuple(v for v in [x, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns
        )


class CollisionForCausalLM(CollisionPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = {r"blocks\.\d+\.attn\.bias"}
    _keys_to_ignore_on_load_missing = set()

    @property
    def all_tied_weights_keys(self):
        return {"lm_head.weight": "token_emb.emb.weight"}

    def __init__(self, config: CollisionConfig):
        super().__init__(config)
        self.config = config
        
        # Direct embedding and transformer layers matching state_dict
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                max_seq_len=config.max_seq_len,
                d_ff=config.d_ff,
                dropout=config.dropout
            ) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.emb.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.token_emb.emb

    def set_input_embeddings(self, value):
        self.token_emb.emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if input_ids.shape[1] > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, "use_return_dict") else True

        if input_ids is not None:
            B, T = input_ids.size()
            x = self.token_emb(input_ids)
            if position_ids is None:
                x = x + self.pos_emb(input_ids)
            else:
                x = x + self.pos_emb.emb(position_ids)
        elif inputs_embeds is not None:
            B, T, _ = inputs_embeds.size()
            x = inputs_embeds
            if position_ids is None:
                positions = torch.arange(T, device=inputs_embeds.device).unsqueeze(0)
                x = x + self.pos_emb.emb(positions)
            else:
                x = x + self.pos_emb.emb(position_ids)
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        x = self.drop(x)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            layer_outputs = block(x, attention_mask=attention_mask, output_attentions=output_attentions)
            x = layer_outputs[0]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        hidden_states = self.ln_f(x)
        logits = self.lm_head(hidden_states)
        logits = torch.nan_to_num(logits, nan=0.0)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        if not return_dict:
            output = (logits,) + (all_hidden_states, all_self_attns)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns
        )


# Compatibility alias
CollisionTransformer = CollisionForCausalLM
