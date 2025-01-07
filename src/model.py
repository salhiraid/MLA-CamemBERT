import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import CamembertConfig

class CamembertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayeNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Debug prints
        # print(f"Embeddings NaN: {torch.isnan(embeddings).any()}")

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
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_x_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Debug query, key, value
        # print(f"Query NaN: {torch.isnan(query_layer).any()}")
        # print(f"Key NaN: {torch.isnan(key_layer).any()}")
        # print(f"Value NaN: {torch.isnan(value_layer).any()}")

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.attention_head_size)

        # Clamp scores to prevent overflow
        attention_scores = torch.clamp(attention_scores, min=-1e9, max=1e9)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) + 1e-9
        attention_probs = self.dropout(attention_probs)

        # Debug attention scores and probabilities
        # print(f"Attention Scores NaN Before Clamp: {torch.isnan(attention_scores).any()}")
        # print(f"Attention Probs NaN: {torch.isnan(attention_probs).any()}")

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.all_head_size)

        # Debug context layer
        # print(f"Context Layer NaN: {torch.isnan(context_layer).any()}")

        return context_layer



class CamembertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = F.gelu if config.hidden_act == "gelu" else nn.ReLU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate

    def forward(self, hidden_states):
        intermediate_output = self.activation(self.dense_1(hidden_states))
        intermediate_output = torch.clamp(intermediate_output, min=-1e9, max=1e9)

        output = self.dense_2(intermediate_output)
        output = self.dropout(output)
        output = self.LayerNorm(output + hidden_states)

        # Debug intermediate and final outputs
        # print(f"Intermediate Output NaN: {torch.isnan(intermediate_output).any()}")
        # print(f"Final Output NaN: {torch.isnan(output).any()}")

        return output


class CamembertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CamembertSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = CamembertFeedForward(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        return self.feed_forward(hidden_states)

class CamembertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CamembertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)

            # Debug prints for each layer
            # print(f"Layer {i} Hidden States NaN: {torch.isnan(hidden_states).any()}")

        return hidden_states

class CamembertLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = F.gelu(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)

        # Debug prints
        # print(f"Logits NaN: {torch.isnan(logits).any()}")

        return logits

class CamembertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CamembertEmbeddings(config)
        self.encoder = CamembertEncoder(config)
        self.head = CamembertLMHead(config) if config.head_type == "MLM" else None

    def forward(self, input_ids, attention_mask=None):
        embedded_input = self.embeddings(input_ids)

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -float('inf')

        encoder_output = self.encoder(embedded_input, attention_mask)
        return self.head(encoder_output)
