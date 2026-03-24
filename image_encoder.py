import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        pretrained_weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet_v3_small_model = models.mobilenet_v3_small(weights=pretrained_weights)

        return_nodes = {"features.12": "last_feature_layer"}

        self.feature_extractor = create_feature_extractor(mobilenet_v3_small_model, return_nodes=return_nodes)

        self.projection = nn.Linear(576, embed_dim)

    def forward(self, images):
        features = self.feature_extractor(images)["last_feature_layer"]

        features = torch.mean(features, dim=(2, 3))

        image_embeds = self.projection(features)

        image_embeds = F.normalize(image_embeds, dim=-1)

        return image_embeds