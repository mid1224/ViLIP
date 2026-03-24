import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext

class TextEncoder(nn.Module):
    def __init__(self, fasttext_model_path, embed_dim=256):
        super().__init__()
    
        self.fasttext_model = fasttext.load_model(fasttext_model_path)

        self.projection = nn.Linear(300, embed_dim)

    def forward(self, captions):
        sentence_vectors = []

        for caption in captions:            
            sentence_vector = torch.from_numpy(self.fasttext_model.get_sentence_vector(caption))
            # torch.from_numpy to convert numpy array to torch tensor

            sentence_vectors.append(sentence_vector)

        sentence_vectors = torch.stack(sentence_vectors).float()

        text_embeds = self.projection(sentence_vectors)

        text_embeds = F.normalize(text_embeds, dim=-1)

        return text_embeds