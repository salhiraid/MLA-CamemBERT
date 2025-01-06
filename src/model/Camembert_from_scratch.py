import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
from packaging import version
from transformers.utils import logging
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CamembertConfig:
    """
    This is the configuration class to store the configuration of a [`CamembertModel`] or a [`TFCamembertModel`]. 
    It defines the model architecture and is used to instantiate a Camembert model with specified arguments.
    """
    
    vocab_size: int = 32005
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-05
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    head_type: str = "MLM"
    classifier_dropout: float = None


class CamembertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)  # Shape (1, max_position_embeddings)
        self.token_type_ids = torch.zeros_like(self.position_ids, dtype=torch.long)  # Shape (1, max_position_embeddings)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length].to(input_ids.device)
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(batch_size, seq_length).to(input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeds + token_type_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    

class CamembertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        batch_size, seq_length, hidden_size = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_size]

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_scores += attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(hidden_states.size(0), -1, self.all_head_size)

        return context_layer
    

class CamembertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CamembertBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CamembertSelfAttention(config)  # Multi-Head Attention
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # LayerNorm après l'attention
        self.mlp = CamembertMLP(config)  # Feed Forward Network (MLP)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # LayerNorm après le MLP

    def forward(self, hidden_states, attention_mask=None):
        # Multi-Head Attention avec résiduel
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + attention_output  # Résiduel
        hidden_states = self.ln_1(hidden_states)  # Normalisation

        # Feed Forward Network (MLP) avec résiduel
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output  # Résiduel
        hidden_states = self.ln_2(hidden_states)  # Normalisation

        return hidden_states


class Camembert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = CamembertEmbeddings(config)
        self.encoder = nn.ModuleList([CamembertBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        for block in self.encoder:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    

model = Camembert(CamembertConfig())