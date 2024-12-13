import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
from packaging import version
from transformers.utils import logging

# Classes to code : 
# Embedding
# Self-attention
# self output 
# Attention output
# CamemBERT block(layer)
# CamemBERT Encoder 
# CamemBERT output
# CamemBERT Model
# We can add the 4 classes to fine-tune the model on the 4 donwsteam tasks
# + one class to load directly a pretrained model

logger = logging.get_logger(__name__)

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
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.attention_head_size)

        # Apply attention mask (broadcast to match attention_scores shape)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_scores += attention_mask

        # Normalize attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of value vectors
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(hidden_states.size(0), -1, self.all_head_size)

        return context_layer

class CamembertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.ReLU()  # Simpler activation for beginners
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return self.dropout(x)

class CamembertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CamembertSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = CamembertFeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention with skip connection
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_norm(hidden_states + attention_output)  # Skip connection

        # Feed-Forward with skip connection
        feed_forward_output = self.feed_forward(attention_output)
        layer_output = self.feed_forward_norm(attention_output + feed_forward_output)  # Skip connection

        return layer_output

class CamembertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CamembertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
    
class CamembertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CamembertEmbeddings(config)  # Ensure it includes word, position, and type embeddings
        self.encoder = CamembertEncoder(config)  # Ensure num_hidden_layers matches config

    def forward(self, input_ids, attention_mask=None):
        # Embedding layer
        embedded_input = self.embeddings(input_ids)

        # Attention mask preparation
        if attention_mask is not None:
            # attention_mask = (1.0 - attention_mask) * -10000.0  # Convert to large negative values for masked positions
            attention_mask = (1.0 - attention_mask) * -float('inf')


        # Encoder
        encoder_output = self.encoder(embedded_input, attention_mask)
        return encoder_output