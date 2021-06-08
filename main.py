import torch
from audio_tf_pytorch import Attention

attn = Attention(
    heads=8,
    dim=512,
    hidden_dim=200
)

x = torch.rand(32, 10, 512)
y = attn(x)

print (y.shape)