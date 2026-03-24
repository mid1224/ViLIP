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

def preprocess_image(image_path):
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406], 
			std=[0.229, 0.224, 0.225]
		),
	])

	image = Image.open(image_path).convert('RGB')
	image = transform(image).unsqueeze(0) # Unsqueeze to add batch dimension

	return image

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
	image_path = "dataset/test/images/000000009895.jpg"
	captions = [
		"Ở trên sân , một cầu thủ đánh bóng đang vung gậy để đánh bóng .",
		"Người đàn ông cầm vợt tennis đang đuổi theo bóng .",
		"Đứa trẻ đang đeo găng tay bóng chày bắt bóng thấp trên sân .",
		"Một cậu bé đang vung gậy đánh bóng và một cậu bé ngồi trên xe đạp .",
	]
	captions = [preprocess_caption(c) for c in captions]

	image = preprocess_image(image_path)

	# Calculate similarity scores and use softmax to get probabilities
	with torch.no_grad():
		image_embed = model.image_encoder(image)
		text_embeds = model.text_encoder(captions)
		logits = image_embed @ text_embeds.T * model.temp.exp() + model.bias
		probs = logits.softmax(dim=1).squeeze(0).tolist()

	# Print results
	print("\nResult:")
	for i in range(len(captions)):
		print(f"{probs[i]:.2%}: {captions[i]}")

if __name__ == "__main__":
	main()
