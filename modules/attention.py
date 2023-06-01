import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearLayer(nn.Module):
    def __init__(
        self, 
        q_dim,
        k_dim,
        layer_num=1,
        embed_dim=768,
        att_heads=6
    ):
        super(BilinearLayer, self).__init__()
        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([]) 
        self.dropout_layers = nn.ModuleList([]) 
        for i in range(layer_num):
            attention_layer = BilinearAttention(
                q_dim = q_dim, 
                k_dim = k_dim, 
                embed_dim=embed_dim, 
                att_heads=att_heads
            )
            self.layers.append(attention_layer)
            self.dropout_layers.append(nn.Dropout(0.5))
            if i<layer_num-1: 
                self.bifeat_emb.append(nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ))
                self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        self.proj = nn.Linear(embed_dim * layer_num + q_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, q_feat, k_feats, q_mask, k_mask):
        k_feats = k_feats.contiguous()
        q_feat = torch.sum(q_feat * q_mask.unsqueeze(-1), 1) / torch.sum(q_mask.unsqueeze(-1), 1)
        
        feat_arr = [q_feat]
        for i, layer in enumerate(self.layers):
            q_feat = layer(q_feat, k_feats, k_mask, q_feat, k_feats)
            q_feat = self.dropout_layers[i](q_feat)
            feat_arr.append(q_feat)
            if i<len(self.layers)-1:
                k_feats_cat = torch.cat([q_feat.unsqueeze(1).expand_as(k_feats), k_feats], dim = -1)
                k_feats = self.bifeat_emb[i](k_feats_cat) + k_feats
                k_feats = self.layer_norms[i](k_feats)

        q_feat = torch.cat(feat_arr, dim=-1)
        q_feat = self.proj(q_feat)
        q_feat = self.layer_norm(q_feat)
        return q_feat

class BilinearAttention(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim=1024, att_heads=8):
        super(BilinearAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = embed_dim

        sequential = []
        sequential.append(nn.Linear(q_dim, output_dim))
        sequential.append(nn.Tanh())
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(k_dim, output_dim))
        sequential.append(nn.Tanh())
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(q_dim, output_dim))
        sequential.append(nn.Tanh())
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(k_dim, output_dim))
        sequential.append(nn.Tanh())
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = Attention(embed_dim//att_heads, embed_dim//att_heads//2)

    def forward(self, query, key, mask, value1, value2):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])
        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

class Attention(nn.Module):
    def __init__(self, dim1=128, dim2=64):
        super(Attention, self).__init__()
        sequential = []
        sequential.append(nn.Linear(dim1, dim2))
        sequential.append(nn.ReLU())
        sequential.append(nn.Dropout(0.1))
        self.attention_last = nn.Linear(dim2, 1)
        self.attention_basic = nn.Sequential(*sequential)
        self.attention_last2 = nn.Linear(dim2, dim1)

    def forward(self, att_map, att_mask, value1, value2):
        att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4:
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn