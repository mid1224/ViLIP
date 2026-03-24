from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import random

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    
    return image

# Custom dataset class follow the torch.utils.data.Dataset format 
# so it can be loaded by torch.utils.data.DataLoader.
class ViIC(Dataset):
    def __init__(self, captions_file):
        # Dictionary to store image paths and their corresponding captions
        self.image_to_captions = defaultdict(list)
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    img_path, caption = parts
                    self.image_to_captions[img_path].append(caption)

        # List to store unique image paths          
        self.image_paths = list(self.image_to_captions.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Look up image path and its captions
        img_path = self.image_paths[index]
        captions = self.image_to_captions[img_path]
        
        # Load and preprocess the image
        image = preprocess_image(img_path)
        
        # Select a caption randomly from the list of captions for this image
        caption = random.choice(captions)
        
        # Return the tensor image and the preprocessed text string
        return image, caption


def get_dataloader(captions_file_path, batch_size=8):
    dataset = ViIC(
        captions_file=captions_file_path
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        drop_last=True, 
        shuffle=True
    )