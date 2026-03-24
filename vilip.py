import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class ViLIP(nn.Module):
    def __init__(self, fasttext_model_path, embed_dim=256, init_temp=1.0, init_bias=0.0):
        super().__init__()

        self.text_encoder = TextEncoder(fasttext_model_path, embed_dim)
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * init_temp)
        self.bias = nn.Parameter(torch.ones([]) * init_bias)

    def forward(self, images, captions):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(captions)

        logits = image_embeds @ text_embeds.T * self.temp.exp() + self.bias

        return logits