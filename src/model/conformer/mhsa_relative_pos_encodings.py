import math

import torch
from torch import nn


class MHSAWithRelativePosEncoding(nn.Module):
    def __init__(self, encoder_dim, max_length, attention_heads=4, dropout_rate=0.0):
        super().__init__()

        assert (
            encoder_dim % attention_heads == 0
        ), "encoder_dim should be divisible by attention_heads"

        self.max_length = max_length

        self.encoder_dim = encoder_dim
        self.attention_heads = attention_heads
        self.head_dim = encoder_dim // attention_heads
        self.register_buffer("sqrt_head_dim", torch.sqrt(torch.tensor(self.head_dim)))

        self.key_mat = nn.Linear(encoder_dim, encoder_dim)
        self.query_mat = nn.Linear(encoder_dim, encoder_dim)
        self.value_mat = nn.Linear(encoder_dim, encoder_dim)
        self.pos_mat = nn.Linear(encoder_dim, encoder_dim, bias=False)

        pe_matrix = self._generate_pe_matrix(max_length, encoder_dim)
        self.register_buffer("pe_matrix", pe_matrix)

        self.u_vec = nn.Parameter(torch.zeros(attention_heads, self.head_dim))
        self.v_vec = nn.Parameter(torch.zeros(attention_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout_rate)

        self.out_proj = nn.Linear(encoder_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.key_mat.weight)
        if self.key_mat.bias is not None:
            nn.init.zeros_(self.key_mat.bias)

        nn.init.kaiming_uniform_(self.query_mat.weight)
        if self.query_mat.bias is not None:
            nn.init.zeros_(self.query_mat.bias)

        nn.init.kaiming_uniform_(self.value_mat.weight)
        if self.value_mat.bias is not None:
            nn.init.zeros_(self.value_mat.bias)

        nn.init.kaiming_uniform_(self.pos_mat.weight)
        if self.pos_mat.bias is not None:
            nn.init.zeros_(self.pos_mat.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _generate_pe_matrix(max_length, dim):
        with torch.no_grad():
            position = torch.arange(-max_length, max_length + 1).float().unsqueeze(1)
            dim_div = 10000 ** (torch.arange(0, dim, 2) / dim)
            pe_mat = torch.zeros(max_length * 2 + 1, dim)
            pe_mat[:, 0::2] = torch.sin(position / dim_div)
            pe_mat[:, 1::2] = torch.cos(position / dim_div)

        return pe_mat

    def forward(self, x):
        seq_len = x.size(1)
        min_relev_pos = self.max_length - seq_len + 1
        max_relev_pos = self.max_length + seq_len - 1
        pos_embedding = self.pe_matrix[min_relev_pos : max_relev_pos + 1].unsqueeze(0)

        query = self.query_mat(x).unflatten(-1, [self.attention_heads, self.head_dim])
        value = (
            self.value_mat(x)
            .unflatten(-1, [self.attention_heads, self.head_dim])
            .transpose(1, 2)
        )
        key = (
            self.key_mat(x)
            .unflatten(-1, [self.attention_heads, self.head_dim])
            .transpose(1, 2)
        )
        pos_key = (
            self.pos_mat(pos_embedding)
            .unflatten(-1, [self.attention_heads, self.head_dim])
            .transpose(1, 2)
        )

        content = torch.matmul(
            (query + self.u_vec).transpose(1, 2), key.transpose(2, 3)
        )
        pos = torch.matmul(
            (query + self.v_vec).transpose(1, 2), pos_key.transpose(2, 3)
        )

        pos = self._shift_relative_pos(pos)[..., : content.size(-1)]
        scores = content + pos

        attention = nn.functional.softmax(scores / self.sqrt_head_dim, dim=-1)
        attention = self.dropout(attention)

        attention = torch.matmul(attention, value).transpose(1, 2)
        attention = attention.contiguous().view(attention.size(0), -1, self.encoder_dim)

        res = self.out_proj(attention)
        return res

    def _shift_relative_pos(self, x):
        batch_size, attention_heads, i, j = x.size()

        zero_pad = torch.zeros(batch_size, attention_heads, i, 1, device=x.device)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(batch_size, attention_heads, j + 1, i)
        x = x_padded[:, :, 1:].view_as(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, encoder_dim, max_length, attention_heads=4, dropout_rate=0.1):
        super().__init__()

        self.mhsa = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            MHSAWithRelativePosEncoding(
                encoder_dim, max_length, attention_heads, dropout_rate=dropout_rate
            ),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.mhsa(x)
