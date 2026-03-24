import re

import torch
from PIL import Image
from torchvision import transforms

from vilip import ViLIP
from underthesea import text_normalize, word_tokenize


def preprocess_caption(text):
	text = text_normalize(text)
	text = word_tokenize(text, format="text")
	text = text.lower()
	text = re.sub(r"[^\w\s_]", " ", text, flags=re.UNICODE)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def main():
	device = "cpu"

	# Load model
	model = ViLIP(
		fasttext_model_path="pretrained/cc.vi.300.bin",
		embed_dim=256,
	)
	model.load_state_dict(torch.load("models/vilip.pt", map_location=device))
	model.to(device)
	model.eval()

	# Prepare data

	# Calculate similarity scores

	# Print results


if __name__ == "__main__":
	main()
