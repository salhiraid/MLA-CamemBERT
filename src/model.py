import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
from packaging import version
from transformers.utils import logging
import torch.nn.functional as F

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
    
        # Embed tokens, positions, and token types
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        print("input_ids shape:", input_ids.shape)
        print("input_ids min:", input_ids.min().item())
        print("input_ids max:", input_ids.max().item())
        print("embedding vocab size:", self.embeddings.num_embeddings)

        # Expand position embeddings to match batch size
        position_embeds = position_embeds.expand(batch_size, -1, -1)
    
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        print("inputs_embeds:", inputs_embeds.shape)
        print("position_embeds:", position_embeds.shape)
        print("token_type_embeds:", token_type_embeds.shape)
        print("inputs_embeds device:", inputs_embeds.device)
        print("position_embeds device:", position_embeds.device)
        print("token_type_embeds device:", token_type_embeds.device)
        print("inputs_embeds dtype:", inputs_embeds.dtype)
        print("position_embeds dtype:", position_embeds.dtype)
        print("token_type_embeds dtype:", token_type_embeds.dtype)
        print("inputs_embeds has Inf:", torch.isinf(inputs_embeds).any())
        
        print("position_embeds has NaN:", torch.isnan(position_embeds).any())
        print("position_embeds has Inf:", torch.isinf(position_embeds).any())
        
        print("token_type_embeds has NaN:", torch.isnan(token_type_embeds).any())
        print("token_type_embeds has Inf:", torch.isinf(token_type_embeds).any())
        print("inputs_embeds has NaN:", torch.isnan(inputs_embeds).any().item())
        print("inputs_embeds has Inf:", torch.isinf(inputs_embeds).any().item())
        
        print("position_embeds has NaN:", torch.isnan(position_embeds).any().item())
        print("position_embeds has Inf:", torch.isinf(position_embeds).any().item())
        
        print("token_type_embeds has NaN:", torch.isnan(token_type_embeds).any().item())
        print("token_type_embeds has Inf:", torch.isinf(token_type_embeds).any().item())
        print("input_ids range:", input_ids.min().item(), input_ids.max().item())
        print("token_type_ids range:", token_type_ids.min().item(), token_type_ids.max().item())
        print("position_ids range:", position_ids.min().item(), position_ids.max().item())
        print("Embedding weights stats:")
        print("word_embeddings mean:", self.word_embeddings.weight.mean().item())
        print("position_embeddings mean:", self.position_embeddings.weight.mean().item())
        print("token_type_embeddings mean:", self.token_type_embeddings.weight.mean().item())




        embeddings = inputs_embeds + position_embeds 
        embeddings =+ token_type_embeds
    
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

class CamembertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dense_2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense_3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.layer_norm(x)
        x - self.dropout1(x)
        x = self.dense_2(x)
        x = self.activation(x)
        x = self.dense_3(x)
        x = self.layer_norm(x)
        x = self.dropout2(x)
        
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
        self.config = config
        self.embeddings = CamembertEmbeddings(config)  
        self.encoder = CamembertEncoder(config)  
        if config.head_type == "MLM":
            self.head = CamembertLMHead(config)
        else:
            raise ValueError(f"Head type {config.head_type} not supported")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embedding layer
        embedded_input = self.embeddings(input_ids)

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -float('inf')

        encoder_output = self.encoder(embedded_input, attention_mask)
        logits = self.head(encoder_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, logits) if labels is not None else logits


class CamembertLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits

