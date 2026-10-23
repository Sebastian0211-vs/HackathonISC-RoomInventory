"""
DINOv2 rewritten for LiteRT's mobile GPU (used by export_embedder.py and export_embedder_probe.py).

On the Galaxy Z Fold4's GPU (LiteRT 2 CompiledModel), the stock graph ([1, tokens, channels] tensors, linear layers
as FULLY_CONNECTED) is right for the CLS token (row 0) but wrong for every other token from the first layer on, so
two unrelated pictures get the same descriptor. Mobile GPUs handle images best, so here the tokens are laid out like
an image row, [1, channels, 1, tokens], every linear layer is a 1x1 convolution, LayerNorm is written out over the
channels and the pooling is a mean over the row. The math is unchanged (checked against Hugging Face's model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(linear):
    c = nn.Conv2d(linear.in_features, linear.out_features, 1)
    c.weight.data = linear.weight.data.clone().view(linear.out_features, linear.in_features, 1, 1)
    c.bias.data = linear.bias.data.clone()
    return c


class ChannelNorm(nn.Module):
    """LayerNorm over the channels of [1, C, 1, N]."""

    def __init__(self, ln):
        super().__init__()
        self.eps = ln.eps
        self.register_buffer("w", ln.weight.data.clone().view(1, -1, 1, 1))
        self.register_buffer("b", ln.bias.data.clone().view(1, -1, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps) * self.w + self.b


class Layer(nn.Module):
    """One Dinov2Layer on [1, C, 1, N]."""

    def __init__(self, hf_layer, heads):
        super().__init__()
        a = hf_layer.attention
        self.heads = heads
        self.norm1, self.norm2 = ChannelNorm(hf_layer.norm1), ChannelNorm(hf_layer.norm2)
        if hasattr(a, "q_proj"):  # transformers 5
            q, k, v, o = a.q_proj, a.k_proj, a.v_proj, a.o_proj
        else:  # transformers 4
            q, k, v, o = a.attention.query, a.attention.key, a.attention.value, a.output.dense
        self.q, self.k, self.v, self.proj = conv1x1(q), conv1x1(k), conv1x1(v), conv1x1(o)
        self.register_buffer("ls1", hf_layer.layer_scale1.lambda1.data.clone().view(1, -1, 1, 1))
        self.register_buffer("ls2", hf_layer.layer_scale2.lambda1.data.clone().view(1, -1, 1, 1))
        self.fc1, self.fc2 = conv1x1(hf_layer.mlp.fc1), conv1x1(hf_layer.mlp.fc2)

    def forward(self, x):  # [1, C, 1, N]
        c, n = x.shape[1], x.shape[3]
        hd = c // self.heads
        y = self.norm1(x)
        q = self.q(y).view(self.heads, hd, n).transpose(1, 2)  # [heads, N, hd]
        k = self.k(y).view(self.heads, hd, n)                  # [heads, hd, N]
        v = self.v(y).view(self.heads, hd, n).transpose(1, 2)  # [heads, N, hd]
        s = torch.matmul(q * (hd ** -0.5), k)                  # [heads, N, N]
        att = torch.softmax(s - s.amax(-1, keepdim=True), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(1, c, 1, n)
        x = x + self.ls1 * self.proj(y)
        y = self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x + self.ls2 * y


class Dinov2Gpu(nn.Module):
    """[1, S, S, 3] -> normalise(normalise(CLS) + normalise(mean of patch tokens)), like the app's embedder."""

    def __init__(self, dino, size):
        super().__init__()
        emb = dino.embeddings
        grid = size // emb.patch_size
        self.n = 1 + grid * grid
        with torch.no_grad():
            pos = emb.interpolate_pos_encoding(torch.zeros(1, self.n, emb.cls_token.shape[-1]), size, size)
        self.register_buffer("pos", pos[0].t().contiguous().view(1, -1, 1, self.n))       # [1, C, 1, N]
        self.register_buffer("cls", emb.cls_token.detach().view(1, -1, 1, 1).clone())     # [1, C, 1, 1]
        self.patch = emb.patch_embeddings.projection
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3))
        heads = dino.config.num_attention_heads
        self.layers = nn.ModuleList(Layer(l, heads) for l in dino.encoder.layer)
        self.final = ChannelNorm(dino.layernorm)

    def hidden(self, x):  # x: [1, S, S, 3] RGB 0..1
        x = ((x - self.mean) / self.std).permute(0, 3, 1, 2)
        p = self.patch(x)                                  # [1, C, g, g]
        p = p.reshape(1, p.shape[1], 1, -1)                # [1, C, 1, g*g]
        h = torch.cat([self.cls, p], 3) + self.pos         # [1, C, 1, N]
        out = [h]
        for layer in self.layers:
            h = layer(h)
            out.append(h)
        return out

    def norm(self, h):
        return self.final(h)

    @staticmethod
    def pool(h):  # -> CLS [1, C] and patch-token mean [1, C]
        return h[:, :, 0, 0], h[:, :, :, 1:].mean(3)[:, :, 0]

    def forward(self, x):
        cls, mean = self.pool(self.norm(self.hidden(x)[-1]))
        e = F.normalize(cls, dim=-1) + F.normalize(mean, dim=-1)
        return F.normalize(e, dim=-1)  # [1, C]
