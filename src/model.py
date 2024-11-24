import torch
import torch.nn as nn
import math

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


class SelfAttention(nn.module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_context_shape)